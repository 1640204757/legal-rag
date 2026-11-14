"""查询扩展的事实保护、失败回退和候选去重测试；不访问网络。"""
import pytest
from local_rag.query_expansion import parse_terms, merge_hits, expand_query


def test_reject_invented_fact():
    with pytest.raises(ValueError):
        parse_terms('老板不给钱', '{"terms":["上海","不给钱"]}')


def test_exact_spans_only():
    assert parse_terms('老板不给钱', '{"terms":["不给钱"]}') == ['不给钱']


def test_merge_unique_and_keep_original():
    a = [{'id':str(i)} for i in range(4)]
    b = [{'id':'0'}, {'id':'4'}, {'id':'5'}]
    result = merge_hits(a, b)
    assert result[:2] == a[:2]
    assert len(result) == 4 and len({x['id'] for x in result}) == 4


def test_no_model_claims_on_failure():
    import asyncio
    class Broken:
        async def chat(self, messages):
            raise RuntimeError('offline')
    result = asyncio.run(expand_query(Broken(), '老板不给钱'))
    assert result['query'] == '老板不给钱'
    assert result['accepted'] is False


def test_synonyms_do_not_replace_original():
    import asyncio
    class Model:
        async def chat(self, messages):
            return '{"terms":["不给钱"]}'
    result = asyncio.run(expand_query(Model(), '老板不给钱'))
    assert result['query'].startswith('老板不给钱')
    assert '劳动关系已成立' not in result['query']
