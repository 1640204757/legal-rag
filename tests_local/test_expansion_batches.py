"""增量发布边界：批次预算、不同来源同名去重、原件与重解析一致性。"""
import pytest
from local_rag.expansion_batches import group_records, verify_cached_baseline


def test_batches_keep_documents_whole_and_skip_unparsed():
    rows = [dict(title='甲', articles=4, status='PARSED'), dict(title='乙', articles=6, status='PARSED'),
            dict(title='丙', articles=3, status='PARSED'), dict(title='隔离', articles=1, status='QUARANTINED')]
    groups = group_records(rows, 8)
    assert [[r['title'] for r in g] for g in groups] == [['甲'], ['乙'], ['丙']]


def test_same_title_different_sources_are_not_both_published():
    rows = [dict(title='甲', articles=4, status='PARSED'), dict(title='甲', articles=4, status='PARSED')]
    assert sum(map(len, group_records(rows, 10))) == 1


def test_nonpositive_budget_rejected():
    with pytest.raises(ValueError):
        group_records([], 0)


def test_new_vectors_cannot_mask_missing_old_cache():
    entries = {'old-1': [1.0]*1024, 'new-1': [1.0]*1024, 'new-2': [1.0]*1024}
    with pytest.raises(ValueError):
        verify_cached_baseline(entries, ['old-1', 'old-2'])
