"""口语改写实验的隔离与留证测试；使用假模型，不访问服务。"""
import asyncio
import pytest

from scripts.experiment_query_rewrite import select_dev, rewrite_once, compare_records


def test_sampling_is_dev_only_and_independent_of_answers():
    rows = [dict(id=f'D{i:02}', split='dev', query=str(i)) for i in range(8)]
    mixed = [dict(id='H', split='holdout', query='不能送模')] + list(reversed(rows))
    assert [q['id'] for q in select_dev(mixed, 4)] == ['D00', 'D02', 'D04', 'D06']
    with pytest.raises(ValueError):
        select_dev(rows, 9)


def test_model_receives_only_query_and_valid_json_is_not_fact_approval():
    class Model:
        async def chat(self, messages):
            assert messages[-1] == {'role': 'user', 'content': '他没给钱'}
            assert len(messages) == 2
            return '{"rewrite":"他尚未付款。"}'
    row = asyncio.run(rewrite_once(Model(), '他没给钱'))
    assert row['candidate_query'] == '他尚未付款。'
    assert row['parse_ok'] is True
    assert row['review_status'] == 'pending'
    assert row['raw'] == '{"rewrite":"他尚未付款。"}'


@pytest.mark.parametrize('raw', ['不好说', '{"rewrite":""}', '{"rewrite":"问题","answer":"结论"}'])
def test_malformed_output_retains_raw_and_falls_back(raw):
    class Model:
        async def chat(self, messages):
            return raw
    row = asyncio.run(rewrite_once(Model(), '原话'))
    assert row['candidate_query'] == '原话'
    assert row['raw'] == raw
    assert not row['parse_ok']


def test_network_failure_stays_in_denominator():
    class Model:
        async def chat(self, messages):
            raise RuntimeError('offline')
    row = asyncio.run(rewrite_once(Model(), '原话'))
    assert row['candidate_query'] == '原话'
    assert row['error'] == 'RuntimeError'


def test_comparison_reports_regressions_not_only_net_gain():
    before = [dict(id='a', top4_rank=1, top8_rank=1), dict(id='b', top4_rank=None, top8_rank=None)]
    after = [dict(id='a', top4_rank=None, top8_rank=None), dict(id='b', top4_rank=2, top8_rank=2)]
    result = compare_records(before, after)
    assert result['top4']['lost'] == ['a']
    assert result['top4']['gained'] == ['b']
    assert result['top4']['before'] == result['top4']['after'] == 1
    with pytest.raises(ValueError):
        compare_records(before, after[:1])
