"""稳定检索复原与地域安全回归；实验断言已归档，不读取40题标注或计入实测成绩。"""
import asyncio
from collections import Counter
from local_rag.retrieval import Retriever, bm25_rank
from local_rag.service import ChatService, uncovered_foreign_region


def test_stable_bm25_does_not_add_title_only_matches():
    rows = [dict(title='测试合同法',text='当事人依法承担责任'),dict(title='其他规范',text='当事人依法承担责任')]
    assert bm25_rank(rows,'合同法') == []


def test_scope_does_not_override_explicit_chinese_law_request():
    assert uncovered_foreign_region('我在德国，现请按中国法律说明合同责任。', []) is None
    assert uncovered_foreign_region('请按德国规则说明通知条件。', [dict(jurisdiction='德国')]) is None


def test_rejecting_chinese_law_is_not_a_request_for_chinese_law():
    assert uncovered_foreign_region('我在德国，不要依据中国法律，请按当地规则说明通知条件。', []) == '德国'


def test_twenty_candidate_request_preserves_neutral_top_four_order():
    class Embed:
        async def embed(self,*args,**kwargs):
            return [[1.,0.]]
    class Store:
        requested = None
        async def search(self,vector,limit):
            self.requested = limit
            return [(i,1.-i*.01) for i in range(6)]
    async def run():
        rows = [dict(title='重复规范' if i<4 else '其他规范'+str(i),text='合同责任',id=str(i)) for i in range(6)]
        store = Store()
        retriever = Retriever(rows,Embed(),store=store)
        retriever.vectors = [[1.,0.]]*6
        hits,mode = await retriever.search('合同责任')
        assert store.requested == 20 and mode == 'hybrid_qdrant'
        assert [r['id'] for r in hits] == ['0','1','2','3']
    asyncio.run(run())


def test_single_law_can_supply_four_relevant_articles_after_rollback():
    async def run():
        rows = [dict(title='单一规范',text='合同责任',id=str(i)) for i in range(4)]
        hits,_ = await Retriever(rows,None).search('合同责任')
        assert len(hits) == 4
    asyncio.run(run())


def test_nonpositive_top_k_returns_no_materials():
    async def run():
        retriever = Retriever([dict(title='规范',text='合同责任',id='1')],None)
        assert (await retriever.search('合同责任',top_k=0))[0] == []
    asyncio.run(run())


def test_foreign_gap_has_safe_final_answer_and_retains_raw_block_evidence():
    class Search:
        rows = [dict(title='中华人民共和国劳动法')]
        async def search(self,*args,**kwargs):
            return [],'hybrid_qdrant'
    class Model:
        async def chat(self,messages):
            return '依据德国劳动法，通知期限通常为两周。\nUSED_IDS: 无'
    async def run():
        service = ChatService(Search(),Model())
        sid = service.create()
        rid = service.start(sid,'请依据德国劳动法解释解雇通知要求。')
        await service.sessions[sid].jobs[rid]['task']
        result = service.result(sid,rid)['result']
        assert result['scope_degraded'] and result['raw_output_blocked']
        assert not result['blocked'] and result['citations'] == []
        assert '德国' in result['answer'] and '未覆盖' in result['answer'] and '无法确认' in result['answer']
        assert '两周' not in result['answer'] and '两周' in result['raw_answer']
        assert any(w['type']=='DEADLINE_VALUE_NOT_IN_CITED_TEXT' for w in result['warnings'])
    asyncio.run(run())
