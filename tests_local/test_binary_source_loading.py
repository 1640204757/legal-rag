"""二进制原件加载边界：只接受登记文件名并对原始字节验哈希。"""
import hashlib
import json
import pytest
from local_rag import ingest


def test_docx_raw_bytes_are_checked_without_fake_html(tmp_path, monkeypatch):
    folder = tmp_path/'example'
    folder.mkdir()
    raw = b'PK-official-original'
    articles = json.dumps([dict(article_no='第一条', locator='正文', text='第一条 示例正文')], ensure_ascii=False).encode()
    record = dict(id='example', status='PARSED', raw_file='source.docx', raw_sha256=hashlib.sha256(raw).hexdigest(),
                  parsed_sha256=hashlib.sha256(articles).hexdigest(), title='示例法', version='2020-01-01', url='https://flk.npc.gov.cn/detail', field='GENERAL')
    (folder/'source.docx').write_bytes(raw)
    (folder/'articles.json').write_bytes(articles)
    (folder/'record.json').write_text(json.dumps(record), encoding='utf-8')
    monkeypatch.setattr(ingest, 'STORE', tmp_path)
    rows, _ = ingest.load_extra()
    assert rows[0]['title'] == '示例法'
    record['raw_file'] = '../outside.docx'
    (folder/'record.json').write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises((ValueError, RuntimeError)):
        ingest.load_extra()
