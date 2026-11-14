"""司法解释收官语料的评测准入适配；只读核验凭据，不改题集和评分。

阅读入口normalize_completion：输入实际COMPLETE凭据及当前corpus/覆盖量，
核对真实JUnit、原件SHA及入库字段后投影成旧准入接口。无网络或文件写入。
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .retrieval import ROOT


def normalize_completion(receipt,corpus_id,coverage):
    """只接受完整第三批收官，不允许把目标数或未完成记录冒充新语料身份。"""
    database=receipt.get('database',{})
    if (receipt.get('stage')!='COMPLETE' or receipt.get('corpus_id')!=corpus_id
            or receipt.get('coverage')!=coverage or database.get('corpus_id')!=corpus_id
            or database.get('points')!=coverage['chunks'] or not database.get('ready')
            or receipt.get('exact_new_payloads_verified')!=receipt.get('added_articles')
            or receipt.get('added_records')!=3 or not receipt.get('repeat_sync_same')):
        raise ValueError('司法解释收官凭据未完成或与当前语料不符')
    attempt=(ROOT/receipt['attempt']).resolve()
    if not attempt.is_relative_to(ROOT/'data/catalogs/judicial-core-20261021/local-batch-003/attempts'):
        raise ValueError('收官运行证据目录不符')
    for key,label in [('before_tests','tests-before-publish'),('final_tests','tests-final')]:
        counts=receipt.get(key,{})
        if counts.get('passed',0)<326 or any(counts.get(k,1)!=0 for k in ['failures','errors','skipped']):
            raise ValueError('收官回归凭据不足')
        suites=list(ET.parse(attempt/(label+'.xml')).getroot().iter('testsuite'))
        actual={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
        actual['passed']=actual['tests']-sum(actual[k] for k in ['failures','errors','skipped'])
        if actual!=counts:
            raise ValueError('收官测试字段与真实JUnit不一致')
    records=receipt.get('records',[])
    if {r.get('catalog_id') for r in records}!={'J07','J14','J15'}:
        raise ValueError('收官记录范围不符')
    for record in records:
        folder=ROOT/'data/catalogs/judicial-core-20261021/local-batch-003/documents'/record['catalog_id']
        path=(folder/record['raw_file']).resolve()
        if not path.is_relative_to(folder.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=record['raw_sha256']:
            raise ValueError('收官原件SHA不符')
    return dict(final=True,pending_titles=[],worker_running=False,latest_verified_coverage=coverage,
                live_database_counts=dict(latest=dict(corpus_id=corpus_id,exact_count=database['points'])),
                evidence_kind='verified_judicial_closeout')
