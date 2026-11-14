"""安全修复回归：期限前缀、声明编号、显式法条及境外规则不得绕过材料校验。"""
import asyncio
import pytest
from local_rag.service import deadline_grounding_issues,ChatService
from prototype.rag.postprocess import postprocess


@pytest.mark.parametrize('answer',[
    '申请时间不得超过30日，依据[M1]。',
    '申请人至少等待三个月，依据[M1]。',
    '应在30天之前提出，依据[M1]。',
    '办理期限为30个工作日，依据[M1]。',
    '至少两个月，依据[M1]。',
    '不得超过一万日，依据[M1]。',
    '至少等待满三个月，依据[M1]。',
])
def test_prefix_deadlines_without_source_are_blocked(answer):
    assert deadline_grounding_issues(answer,[dict(material_id='M1',text='申请需提交证明')])


def test_workdays_are_not_backed_by_calendar_days():
    assert deadline_grounding_issues('期限为30个工作日，依据[M1]。',[dict(material_id='M1',text='30日内提出申请')])


def material(title='中华人民共和国民法典',locator='第五百七十七条',mid='M1'):
    return dict(material_id=mid,title=title,locator=locator,version_id='2020-05-28',
                text='当事人不履行合同义务，应当承担违约责任。',source_url='https://example.test',content_scope='ARTICLES_ONLY')


def test_unknown_used_id_alone_is_blocked():
    assert postprocess('可以请求赔偿。\nUSED_IDS: M99',[material()])['blocking']
    assert postprocess('可以请求赔偿。\nUSED_IDS: 使用M99',[material()])['blocking']


@pytest.mark.parametrize('answer',[
    '依据《中华人民共和国民法典》第九万九千九百九十九条可以赔偿。[M1]\nUSED_IDS: M1',
    '根据民法典第99999条可以赔偿。[M1]\nUSED_IDS: M1',
    '德国《劳动法》第116条规定了通知期限。\nUSED_IDS: 无',
    '《不存在法》规定可以赔偿。[M1]\nUSED_IDS: M1',
    '《不存在法》第577条规定赔偿，参考《民法典》[M1]\nUSED_IDS: M1',
])
def test_unprovided_law_or_article_is_blocked(answer):
    assert postprocess(answer,[material()])['blocking']


def test_named_article_cannot_bind_other_material():
    materials=[material(),material('中华人民共和国劳动法','第十七条','M2')]
    assert postprocess('依据《民法典》第五百七十七条办理。[M2]\nUSED_IDS: M2',materials)['blocking']


def test_supported_article_with_valid_id_is_not_blocked():
    assert not postprocess('依据《民法典》第577条可要求承担责任。[M1]\nUSED_IDS: M1',[material()])['blocking']


def test_foreign_scope_keeps_real_search_but_removes_domestic_materials():
    class Search:
        rows=[dict(field='LABOR',title='中华人民共和国劳动法')]
        calls=0
        async def search(self,query,top_k=4):
            self.calls+=1
            return [dict(material(),kind='LAW_ARTICLE',verified_scope='快照')],'hybrid_qdrant'
    class Model:
        messages=None
        async def chat(self,messages):
            self.messages=messages
            return '## 不确定性\n现有资料未覆盖德国当地法律，无法确认具体期限。\nUSED_IDS: 无'
    async def run():
        search,model=Search(),Model()
        service=ChatService(search,model)
        sid=service.create()
        rid=service.start(sid,'我在德国工作，请依据德国当地法律说明解雇通知期限。')
        await service.sessions[sid].jobs[rid]['task']
        result=service.result(sid,rid)['result']
        assert search.calls==1 and result['retrieved']==[]
        assert '不得输出' in model.messages[0]['content']
        assert '未覆盖' in result['answer']
    asyncio.run(run())
