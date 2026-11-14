"""新语料准入凭据测试，不修改40题及冻结评分函数。"""
from copy import deepcopy
import json

import pytest

from local_rag.eval_judicial_evidence import normalize_completion
from local_rag.retrieval import ROOT


def receipt():
    """只读本机真实第三批完成凭据，反例只在内存修改。"""
    return json.loads((ROOT/'data/catalogs/judicial-core-20261021/local-batch-003/completion.json').read_text(encoding='utf-8'))


def test_completed_judicial_corpus_admitted():
    r=receipt()
    e=normalize_completion(r,r['corpus_id'],r['coverage'])
    assert e['final'] and e['live_database_counts']['latest']['exact_count']==46701


@pytest.mark.parametrize('key',['stage','corpus_id','exact_new_payloads_verified','final_tests'])
def test_incomplete_or_conflicting_evidence_rejected(key):
    original=receipt();r=deepcopy(original)
    r[key]=dict(passed=0,failures=1,errors=0,skipped=0) if key=='final_tests' else 'INVALID'
    with pytest.raises(ValueError):
        normalize_completion(r,original['corpus_id'],original['coverage'])
