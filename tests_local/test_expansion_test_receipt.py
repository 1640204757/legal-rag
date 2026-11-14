"""验收凭据的失败边界：旧语料、部分命令和被替换日志均不能当作完整通过。"""
import hashlib
import json
import pytest
from local_rag.expansion_test_receipt import source_hashes, test_arguments as full_arguments, verify_receipt


@pytest.mark.parametrize('tamper', ['corpus', 'scope', 'exit', 'log', 'source'])
def test_final_receipt_rejects_stale_or_partial_evidence(tmp_path, tamper):
    """构造无网络凭据后改变一项关键事实，最终门槛必须拒绝。"""
    out = tmp_path/'out'
    out.mkdir()
    (tmp_path/'local_rag').mkdir()
    module = tmp_path/'local_rag'/'example.py'
    module.write_text('value = 1', encoding='utf-8')
    (out/'final-tests.txt').write_text('239 passed', encoding='utf-8')
    (out/'final-tests.xml').write_text('<testsuites><testsuite>'+''.join(
        f'<testcase name="case{i}"/>' for i in range(239))+'</testsuite></testsuites>', encoding='utf-8')
    receipt = dict(returncode=0, arguments=full_arguments(), corpus_id='current',
                   source_sha256=source_hashes(tmp_path), passed=239,
                   artifacts_sha256={name: hashlib.sha256((out/name).read_bytes()).hexdigest()
                                     for name in ['final-tests.txt', 'final-tests.xml']})
    if tamper == 'corpus':
        receipt['corpus_id'] = 'old'
    elif tamper == 'scope':
        receipt['arguments'] = ['-m', 'pytest', 'one_test.py']
    elif tamper == 'exit':
        receipt['returncode'] = 1
    elif tamper == 'log':
        (out/'final-tests.txt').write_text('1 passed', encoding='utf-8')
    else:
        module.write_text('value = 2', encoding='utf-8')
    (out/'final-tests.receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError):
        verify_receipt(tmp_path, out, 'current')
