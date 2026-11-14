"""赔偿倍数和确定金额的词面校对。

材料里没有写出的“三倍”“十倍”或具体元数，不能当成已经核对的法律结论。
用户自己说出的金额可以复述。倍数即使用户问到了，也必须出现在所引材料里。
"""
import asyncio

from local_rag.service import ChatService, amount_grounding_issues
from prototype.rag.input_builder import Material


def test_multiple_absent_from_cited_text_is_rejected():
    answer = '可以要求价款的三倍赔偿，依据[M1]。'
    materials = [{'material_id': 'M1', 'text': '经营者应当退还货款。'}]
    assert amount_grounding_issues(answer, materials)


def test_multiple_present_in_cited_text_is_kept():
    answer = '可以要求费用的三倍赔偿，依据[M1]。'
    materials = [{'material_id': 'M1', 'text': '增加赔偿的金额为费用的三倍。'}]
    assert not amount_grounding_issues(answer, materials)


def test_colloquial_multiple_still_needs_the_words_in_the_article():
    answer = '这种情况可以退一赔十，依据[M1]。'
    missing = [{'material_id': 'M1', 'text': '生产者应当赔偿损失。'}]
    present = [{'material_id': 'M1', 'text': '赔偿金不足价款十倍的，为价款十倍。'}]
    assert amount_grounding_issues(answer, missing)
    assert not amount_grounding_issues(answer, present)


def test_arabic_multiple_matches_chinese_words_in_the_article():
    answer = '可以要求3倍赔偿，依据[M1]。'
    materials = [{'material_id': 'M1', 'text': '增加赔偿的金额为价款的三倍。'}]
    assert not amount_grounding_issues(answer, materials)


def test_user_stated_amount_can_be_repeated_but_a_new_amount_cannot():
    question = '这双鞋我付了3000元，商家不承认质量问题。'
    materials = [{'material_id': 'M1', 'text': '经营者提供的商品不符合质量要求的，可以退货。'}]
    assert not amount_grounding_issues('你支付的3000元应先保留凭证，依据[M1]。', materials, question=question)
    assert amount_grounding_issues('商家应当赔偿9000元，依据[M1]。', materials, question=question)


def test_amount_from_another_material_does_not_support_the_wrong_citation():
    materials = [
        {'material_id': 'M1', 'text': '可以退货。'},
        {'material_id': 'M2', 'text': '增加赔偿的金额为费用的三倍。'},
    ]
    assert amount_grounding_issues('依据[M1]，可以要求三倍赔偿。', materials)


def test_service_hides_ungrounded_amount_and_keeps_grounded_one():
    from tests_local.test_answer_permission import reading
    material = Material(
        material_id='M1', kind='LAW_ARTICLE', title='测试规范', locator='第五十五条',
        text='增加赔偿的金额为消费者购买商品的价款的三倍。',
        version_id='测试版本', source_url='https://www.gov.cn/test', verified_scope='测试')

    class Search:
        rows = [material.__dict__]

        async def search(self, query, top_k=4):
            return self.rows, 'lexical'

    class Model:
        raw = ''

        async def chat(self, messages):
            return self.raw

    async def run():
        model = Model()
        service = ChatService(Search(), model, official_catalog=[])
        samples = [
            ('商家应当赔偿9000元，依据[M1]。', True),
            # 数值在材料中仅授权原文阅读，不再授权对用户直接提出赔偿请求。
            (reading(quote=material.text), False),
        ]
        for raw, blocked in samples:
            model.raw = raw
            sid = service.create()
            rid = service.start(sid, '鞋是假的，能要求什么')
            await service.sessions[sid].jobs[rid]['task']
            result = service.result(sid, rid)['result']
            assert result['blocked'] is blocked
            assert result['raw_answer'] == raw
            if blocked:
                assert result['citations'] == []
                assert '9000' not in result['answer']
                assert any(item['type'] == 'AMOUNT_VALUE_NOT_IN_CITED_TEXT' for item in result['warnings'])
            else:
                assert result['citations'] and '三倍' in result['answer']

    asyncio.run(run())
