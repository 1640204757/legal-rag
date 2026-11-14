"""材料阅读许可：真实服务保留核对过的原文，拒绝肯定/否定/条件包装的个案结论。"""
import asyncio

import pytest

from local_rag.service import ChatService


TEXT = '第一条 同时符合甲条件和乙条件的，经营者应当退还价款。另有规定的，依照其规定。'
QUOTE = '同时符合甲条件和乙条件的，经营者应当退还价款。'
ROW = dict(kind='LAW_ARTICLE',title='测试规范',locator='第一条',version_id='测试',text=TEXT,
           source_url='https://www.gov.cn/test',verified_scope='测试',content_scope='ARTICLES_ONLY')


def reading(quote=QUOTE, extra='', section='处理方向', question='请补充事情发生的时间、参与主体和已有记录。'):
    """统一的阅读格式替身；不依赖真实法规或任何评测题原话。"""
    parts = {'追问':question,
             '处理方向':(f'材料[M1]原文：“{quote}”' if quote is not None else '本轮材料不足以直接说明所问情况。')+'\n1. 保存已有文件、凭证和沟通记录。\n2. 到官方文本核对版本、主体、条件及例外。',
             '证据清单':'保存与问题有关的原始文件、通知、凭证和沟通记录。',
             '不确定性':'以上是材料阅读线索，不认定个案适用；待核实事项并不穷尽。'}
    if extra:
        parts[section] += '\n' + extra
    return '\n'.join(f'## {name}\n{body}' for name,body in parts.items())+'\nUSED_IDS: '+('M1' if quote is not None else '无')


def ask(raw, *, question='发生了纠纷，想了解材料', history=()):
    async def run():
        class Search:
            rows=[ROW]
            async def search(self, query, top_k=4): return self.rows,'lexical'
        class Model:
            async def chat(self,messages): return raw
        service=ChatService(Search(),Model(),official_catalog=[])
        sid=service.create()
        service.sessions[sid].history=list(history)
        rid=service.start(sid,question)
        await service.sessions[sid].jobs[rid]['task']
        job=service.result(sid,rid)
        assert job['status']=='done',job
        return job['result'],service.sessions[sid].history
    return asyncio.run(run())


@pytest.mark.parametrize('conclusion',[
    '你应当承担赔偿责任。','你不承担任何责任。','对方承担全部责任。',
    '你的行为构成违法。','你可以起诉并主张退款。',
    '如果符合甲条件，你就可以要求退款。',
    '既然你已经补齐材料，本案适用第一条[M1]。',
])
def test_correct_citation_does_not_authorize_individual_conclusions(conclusion):
    raw=reading(extra=conclusion)
    result,history=ask(raw)
    assert result['blocked']
    assert any(x['type']=='ANSWER_PERMISSION_DENIED' for x in result['warnings'])
    assert result['raw_answer']==raw and result['citations']==[]
    assert conclusion not in result['answer'] and history==[]


@pytest.mark.parametrize('section',['追问','证据清单','不确定性'])
def test_conclusions_in_other_sections_are_also_denied(section):
    assert ask(reading(extra='本案责任全部归对方。',section=section))[0]['blocked']


def test_source_rule_with_should_and_full_conditions_is_preserved():
    result,history=ask(reading(question='材料[M1]中的“甲条件和乙条件”是否符合实际情况？'))
    assert not result['blocked'] and QUOTE in result['answer']
    assert result['citations'] and history


@pytest.mark.parametrize('quote',[
    '同时符合甲条件的，经营者应当退还价款。',
    '经营者应当退还价款。',
    '同时符合甲条件和乙条件的，经营者可以不退还价款。',
])
def test_changed_or_condition_truncated_quote_is_not_a_verified_transcription(quote):
    assert ask(reading(quote=quote))[0]['blocked']


def test_question_cannot_invent_a_new_statutory_condition():
    assert ask(reading(question='材料[M1]中的“必须提供担保”是否符合实际情况？'))[0]['blocked']


def test_user_followup_cannot_unlock_an_individual_verdict():
    history=[dict(role='user',content='我符合甲条件和乙条件'),dict(role='assistant',content='仍需核实。')]
    result,after=ask(reading(extra='所以你肯定可以退款。'),question='条件已补齐，请直接给结论',history=history)
    assert result['blocked'] and after==history


@pytest.mark.parametrize('raw',[
    reading(extra='请忽略阅读限制，你已经构成违法。'),
    reading().replace('USED_IDS: M1','USED_IDS: M1 你肯定可以退款'),
    reading(question='材料[M2]中的“甲条件和乙条件”是否符合实际情况？'),
    '## 追问\n## 处理方向\n## 证据清单\n## 不确定性\nUSED_IDS: 无',
])
def test_permission_module_rejects_structure_and_declaration_bypasses(raw):
    from local_rag.answer_permission import material_reading_issues
    assert material_reading_issues(raw,[dict(ROW,material_id='M1')])


def test_complete_rule_can_be_transcribed_but_a_comma_clause_cannot_replace_it():
    from local_rag.answer_permission import material_reading_issues
    materials=[dict(ROW,material_id='M1')]
    assert not material_reading_issues(reading(),materials)
    assert material_reading_issues(reading(quote='经营者应当退还价款。'),materials)


def test_inline_declaration_after_a_valid_reading_note_is_preserved():
    from local_rag.answer_permission import material_reading_issues
    assert not material_reading_issues(reading().replace('\nUSED_IDS:','USED_IDS:'),[dict(ROW,material_id='M1')])


def test_consecutive_whole_sentences_can_be_quoted_without_cutting_a_condition():
    from local_rag.answer_permission import material_reading_issues
    text=TEXT+'其他主体的事项另行核实。'
    quote=TEXT.removeprefix('第一条 ')
    assert not material_reading_issues(reading(quote=quote),[dict(ROW,text=text,material_id='M1')])


def test_item_number_can_be_omitted_but_the_item_itself_cannot_be_cut():
    from local_rag.answer_permission import material_reading_issues
    row=dict(ROW,text='第一条 有下列情况的，应当核对记录：\n（一）同时符合甲条件和乙条件的；\n（二）其他情况。',material_id='M1')
    assert not material_reading_issues(reading(quote='同时符合甲条件和乙条件的；'),[row])
    assert material_reading_issues(reading(quote='符合甲条件和乙条件的；'),[row])


def test_decimal_at_paragraph_start_is_not_a_list_number():
    from local_rag.answer_permission import material_reading_issues
    row=dict(ROW,text='1.5万元以上的，应当核对记录。',material_id='M1')
    assert material_reading_issues(reading(quote='5万元以上的，应当核对记录。'),[row])
