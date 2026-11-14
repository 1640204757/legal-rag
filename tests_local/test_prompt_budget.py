"""预算尺子回归：观测真实_run，完整选择材料、计入历史，并保证不调用回答模型。"""
import asyncio

from scripts.measure_prompt_budget import aggregate, probe_question
from tests_local.test_display_candidates import article


def test_budget_probe_uses_real_message_budget_and_keeps_whole_articles():
    class Search:
        rows = [dict(article(0), text='第一条 '+ '甲'*1800), article(1), article(2)]
        async def search(self, query, top_k=4):
            return self.rows[:top_k], 'hybrid_qdrant'

    q=dict(query='普通纠纷怎么处理', expect_any=[dict(title='规范1', article_no='第二条')])
    row=asyncio.run(probe_question(Search(),q,6,official_fetcher=None))
    assert not row['assembly_error'] and row['target_sent']
    assert row['main_candidates']==3 and row['byte_budget_dropped']>=1
    assert row['sent_count']==2 and row['material_chars']==sum(len(r['text']) for r in Search.rows[1:])
    assert row['message_estimated_bytes']<=7000
    assert aggregate([row])['total_dropped']==1


def test_budget_probe_counts_history_and_compresses_overlong_history_without_chat():
    class Search:
        rows = [article(0)]
        async def search(self, query, top_k=4): return self.rows, 'hybrid_qdrant'
    q=dict(query='普通纠纷',expect_any=[dict(title='规范0',article_no='第一条')])
    history=[dict(role='user',content='核实情况'),dict(role='assistant',content='请保存记录。')]
    row=asyncio.run(probe_question(Search(),q,5,history=history,official_fetcher=None))
    assert row['history_bytes']==sum(len(m['content'].encode('utf-8')) for m in history)
    assert row['history_messages_sent']==2
    # 超长历史不再让整题失败：装配把旧回答压成一行，材料仍保底，总字节守住硬顶。
    history[1]['content']='过长历史'*2000
    compressed=asyncio.run(probe_question(Search(),q,5,history=history,official_fetcher=None))
    assert not compressed['assembly_error'] and compressed['sent_count']==1
    assert compressed['message_estimated_bytes']<=7000
    assert compressed['history_bytes']<len(history[1]['content'].encode('utf-8'))
    assert aggregate([row,compressed])['assembly_errors']==0
