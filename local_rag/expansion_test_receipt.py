"""本轮完整回归凭据：绑定代码、语料、执行命令及JUnit原始结果。

主入口verify_receipt仅读取本地文件；拒绝旧代码、部分范围、失败或缺失的测试结果。
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def source_hashes(root):
    """对生产模块、测试及本轮脚本计算指纹；文档修改无需重新执行测试。"""
    paths = set()
    for folder in ['local_rag', 'prototype/rag', 'tests_local', 'prototype/tests', 'scripts']:
        paths.update((root/folder).rglob('*.py'))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def test_arguments():
    """固定两个完整测试目录，禁止通过额外选择参数缩小范围。"""
    return ['-m', 'pytest', 'tests_local', 'prototype/tests', '-q', '--rootdir', '.',
            '--junitxml=artifacts/local-rag/expansion-20261021/final-tests.xml']


def verify_receipt(root, out, corpus_id):
    """验证实际运行的完整范围、零退出、源码和语料快照，以及逐例JUnit结果。"""
    receipt = json.loads((out/'final-tests.receipt.json').read_text(encoding='utf-8'))
    if receipt['returncode'] != 0 or receipt['arguments'] != test_arguments():
        raise ValueError('最终测试退出失败或执行范围不完整')
    if receipt['source_sha256'] != source_hashes(root) or receipt['corpus_id'] != corpus_id:
        raise ValueError('最终测试凭据的代码或语料已经改变')
    for name in ['final-tests.txt', 'final-tests.xml']:
        if hashlib.sha256((out/name).read_bytes()).hexdigest() != receipt['artifacts_sha256'][name]:
            raise ValueError('最终测试原始结果哈希不一致')
    cases = ET.parse(out/'final-tests.xml').getroot().findall('.//testcase')
    if len(cases) < 239 or any(c.find(tag) is not None for c in cases for tag in ['failure', 'error', 'skipped']):
        raise ValueError('完整回归缺少原239项或存在失败/跳过')
    if len(cases) != receipt['passed']:
        raise ValueError('逐例结果与测试计数不一致')
    return receipt
