"""新语料评测入口的只读边界与真实Token观测测试，桩数据不计入实测成绩。"""
import pytest
from local_rag import eval_runtime as runtime


def test_changed_corpus_requires_verified_expansion():
    """只有完成批次的真实corpus与覆盖数吻合，才允许保留旧题测新库。"""
    spec = {'corpus_id': 'old'}
    coverage = {'laws': 867, 'chunks': 46246}
    evidence = {'final': True, 'pending_titles': [], 'worker_running': False,
                'latest_verified_coverage': coverage,
                'live_database_counts': {'latest': {'corpus_id': 'new', 'exact_count': 46246}}}
    assert runtime.validate_corpus_scope(spec, 'new', coverage, evidence) == 'verified_expansion'
    for bad in [None, dict(evidence, final=False), dict(evidence, pending_titles=['尚未完成'])]:
        with pytest.raises(ValueError):
            runtime.validate_corpus_scope(spec, 'new', coverage, bad)
    with pytest.raises(ValueError):
        runtime.validate_corpus_scope(spec, 'tampered', coverage, evidence)


def test_tokens_are_taken_from_completed_real_response_only():
    """不从字数估算Token，缺少终止包时明确记为未取得。"""
    event = {'service': 'ollama', 'url': 'http://127.0.0.1:11434/api/chat',
             'response_text': '{"done":true,"prompt_eval_count":123,"eval_count":45}\n'}
    assert runtime.token_usage([event])['prompt_tokens'] == 123
    assert runtime.token_usage([event])['completion_tokens'] == 45
    assert runtime.token_usage([])['prompt_tokens'] is None
