"""真实材料适配器边界测试：伪造的是小型本地夹具，不联网或修改法规库。"""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('prototype.saved_material_adapter'))
        from prototype import saved_material_adapter
        self.api = saved_material_adapter
        parent = Path(__file__).resolve().parents[1] / '.tmp/material-adapter-tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root/'data/library/test-law'
        self.folder.mkdir(parents=True)
        self.article = dict(article_no='第一条', text='第一条 测试条件应当明确。', locator='测试位置')
        self.raw = '<html><body>第一条 测试条件应当明确。</body></html>'.encode()
        self.articles = json.dumps([self.article], ensure_ascii=False).encode()
        self.record = dict(id='test-law', title='测试材料', url='https://example.gov.cn/law',
                           version='2020-01-01', status='PARSED', articles=1,
                           raw_sha256=sha(self.raw), parsed_sha256=sha(self.articles))
        self.write_source()

    def write_source(self):
        (self.folder/'record.json').write_text(json.dumps(self.record), encoding='utf-8')
        (self.folder/'source.html').write_bytes(self.raw)
        (self.folder/'articles.json').write_bytes(self.articles)
        self.bindings = {p.relative_to(self.root).as_posix(): sha(p.read_bytes()) for p in self.folder.iterdir() if p.is_file()}

    def build(self):
        return self.api.build_source_catalog(self.root, 'test-law', self.bindings)

    def test_catalog_comes_from_independent_source_not_candidate(self):
        rows, provenance = self.build()
        self.assertEqual(rows[0]['text'], self.article['text'])
        self.assertEqual(rows[0]['id'], 'test-law/第一条')
        self.assertEqual(provenance['raw_text_match_count'], 1)
        self.assertEqual(provenance['version_applicability'], 'NOT_TESTED')

    def test_record_tampering_rejected_even_with_matching_source_hashes(self):
        self.record['url'] = 'https://changed.gov.cn/law'
        (self.folder/'record.json').write_text(json.dumps(self.record),encoding='utf-8')
        with self.assertRaises(ValueError): self.build()

    def test_raw_and_parsed_bytes_are_both_bound(self):
        for filename in ('source.html', 'articles.json'):
            with self.subTest(filename=filename):
                p=self.folder/filename; original=p.read_bytes(); p.write_bytes(original+b' ')
                with self.assertRaises(ValueError): self.build()
                p.write_bytes(original)

    def test_rehashed_fake_article_still_requires_raw_text(self):
        self.article['text']='第一条 原始网页没有的句子。'
        self.articles=json.dumps([self.article],ensure_ascii=False).encode()
        self.record['parsed_sha256']=sha(self.articles)
        self.write_source()
        with self.assertRaises(ValueError): self.build()

    def test_script_content_cannot_back_an_article(self):
        self.raw='<html><script>第一条 测试条件应当明确。</script><body>其他内容</body></html>'.encode()
        self.record['raw_sha256']=sha(self.raw); self.write_source()
        with self.assertRaises(ValueError): self.build()

    def test_only_whitespace_is_normalized(self):
        self.raw='<html><body>第一条\n测试条件 应当明确。</body></html>'.encode()
        self.record['raw_sha256']=sha(self.raw); self.write_source()
        self.assertEqual(self.build()[1]['raw_text_match_count'],1)
        self.raw='<html><body>第一条 测试条件不应当明确。</body></html>'.encode()
        self.record['raw_sha256']=sha(self.raw); self.write_source()
        with self.assertRaises(ValueError): self.build()

    def test_raw_filename_and_source_identifier_cannot_escape(self):
        self.record['raw_file']='../source.html'; self.write_source()
        with self.assertRaises(ValueError): self.build()
        for sid in ('../test-law','..','/absolute','test/law'):
            with self.subTest(sid=sid), self.assertRaises(ValueError):
                self.api.build_source_catalog(self.root,sid,self.bindings)

    def test_duplicate_article_numbers_refused(self):
        self.articles=json.dumps([self.article,self.article],ensure_ascii=False).encode()
        self.record.update(articles=2,parsed_sha256=sha(self.articles)); self.write_source()
        with self.assertRaises(ValueError): self.build()

    def test_docx_plain_text_read_without_office_or_network(self):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as z:
            z.writestr('word/document.xml','<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>第一条 测试条件应当明确。</w:t></w:r></w:p></w:body></w:document>')
        self.record.update(raw_file='source.docx',raw_sha256=sha(buffer.getvalue()))
        (self.folder/'source.docx').write_bytes(buffer.getvalue()); self.write_source()
        self.assertEqual(self.build()[1]['raw_format'],'source.docx')

    def test_trace_order_and_complete_snapshot_are_preserved(self):
        rows,_=self.build()
        trace=dict(case_id='Q01',query='真实格式测试',a_ordered_ids=[rows[0]['id']],
                   k4=[rows[0]['id']],k8=[rows[0]['id']],fused_pool=[{'id':rows[0]['id'],'score':.2}],complete=True)
        packet=self.api.adapt_trace(trace, rows)
        self.assertEqual(packet['candidates'],rows)
        self.assertEqual(packet['branch_trace'],trace)
        self.assertEqual(packet['citations'],[])
        self.assertEqual(packet['actions'],[])
        trace['k4']=[]
        with self.assertRaises(ValueError): self.api.adapt_trace(trace,rows)

    def test_candidate_modified_in_corpus_does_not_change_independent_catalog(self):
        from prototype.evidence_sidecar import check_evidence
        rows,_=self.build(); fake=deepcopy(rows); fake[0]['text']='篡改候选'
        packet=dict(question='测试',candidates=fake,citations=[],semantic_observations=[],actions=[])
        self.assertEqual(check_evidence(packet,rows)['identity'],'FAIL')

    def test_official_url_and_record_identity_required(self):
        for field,value in (('id','wrong-law'),('url','https://example.com/a'),('status','UNAVAILABLE')):
            with self.subTest(field=field):
                original=self.record[field]; self.record[field]=value; self.write_source()
                with self.assertRaises(ValueError): self.build()
                self.record[field]=original; self.write_source()


if __name__=='__main__':
    unittest.main()
