"""简称/指代放行与反例：不允许借同号条文、跨版本或虚构材料绕过校验。"""
import json
from pathlib import Path

import pytest

from prototype.rag.grounding import article_grounding_issues
from prototype.rag.postprocess import postprocess,parse_used_ids
from local_rag.evaluation import judge_degrade,audit_citations


def material(mid,title,article,version='2026'):
    """合成材料只用于结构校验，不是新法规或真实问答成绩。"""
    return dict(material_id=mid,title=title,locator=article,version_id=version,text=article+' 示例正文。',kind='LAW_ARTICLE')


@pytest.mark.parametrize('ident',['Q02','Q05','Q08'])
def test_actual_valid_name_bindings_from_saved_answers(ident):
    # 私有历史报告不随源码发布；保留有证据时的额外核对，不让全新检出依赖私人回答。
    report=Path(__file__).resolve().parents[1]/'artifacts/local-rag/eval_report_20261022_judicial_closeout.json'
    if not report.is_file():
        pytest.skip('本机历史回答报告未提供；公开合成材料回归照常运行')
    r=json.loads(report.read_text(encoding='utf-8'))
    q=next(q for q in r['questions'] if q['id']==ident)
    assert article_grounding_issues(q['raw_answer'],q['assembled_materials'])==[]


@pytest.mark.parametrize('answer,mats',[
    ('劳动合同法第三十八条[M1]、劳动法第四十四条[M2]。',
     [material('M1','中华人民共和国劳动合同法','第三十八条'),material('M2','中华人民共和国劳动法','第四十四条')]),
    ('民法典第五百八十五条[M1]和本解释第六十五条[M2]。',
     [material('M1','中华人民共和国民法典','第五百八十五条'),
      material('M2','最高人民法院关于适用《中华人民共和国民法典》合同编通则若干问题的解释','第六十五条')]),
    ('《中华人民共和国消费者权益保护法》第二十四条[M1]。',
     [material('M1','中华人民共和国消费者权益保护法','第二十四条')]),
])
def test_public_synthetic_name_bindings_without_private_reports(answer,mats):
    """公开合成引用覆盖简称、解释指代、完整法名；不复述历史用户的问题或回答。"""
    assert article_grounding_issues(answer,mats)==[]


@pytest.mark.parametrize('alias,title',[('民诉法','中华人民共和国民事诉讼法'),('刑诉法','中华人民共和国刑事诉讼法'),('劳动法','中华人民共和国劳动法'),('劳动合同法','中华人民共和国劳动合同法')])
def test_statute_alias(alias,title):
    assert not article_grounding_issues(alias+'第十条[M1]。',[material('M1',title,'第十条')])


def test_generic_interpretation_ambiguous_without_ids_is_blocked():
    mats=[material('M1','最高人民法院关于某甲的解释','第五条'),material('M2','最高人民法院关于某乙的解释','第五条')]
    assert article_grounding_issues('解释第五条。',mats)
    assert not article_grounding_issues('本解释第五条[M2]。',mats)
    mats[1]['title']=mats[0]['title']
    mats[1]['version_id']='2020'
    assert article_grounding_issues('解释第五条。',mats)


@pytest.mark.parametrize('answer',['德国民法典第五条[M1]。','《未知法》第五条[M1]。','劳动法第五条[M1]。','民法典第九万九千九百九十九条[M1]。'])
def test_wrong_law_foreign_or_fake_article_still_blocked(answer):
    assert article_grounding_issues(answer,[material('M1','中华人民共和国民法典','第五条')])


@pytest.mark.parametrize('separator',[', ',' ','，','、'])
def test_used_ids_separators_and_unknown_id(separator):
    mats=[material('M1','中华人民共和国民法典','第五条')]
    assert parse_used_ids('USED_IDS: M1'+separator+'M99')==['M1','M99']
    assert postprocess('依据M1处理。\nUSED_IDS: M1'+separator+'M99',mats)['blocking']


def test_general_missing_material_degrade_expression():
    assert judge_degrade('南山区无相关材料，无法提供具体细则。',False,['南山区'])
    assert not judge_degrade('南山区无相关材料，但补贴为3000元。',False,['南山区'])


def test_independent_audit_accepts_unambiguous_interpretation_alias():
    mats=[material('M1','中华人民共和国民法典','第五百八十五条'),material('M2','最高人民法院关于适用《中华人民共和国民法典》合同编通则若干问题的解释','第六十五条')]
    result=audit_citations('民法典第五百八十五条和解释第六十五条[M1][M2]。',mats,mats)
    assert not result['false_citation']


def test_independent_audit_rejects_foreign_modifier():
    mats=[material('M1','中华人民共和国民法典','第五条')]
    assert audit_citations('德国民法典第五条[M1]。',mats,mats)['false_citation']


def test_adjacent_swapped_ids_do_not_borrow_other_reference_in_sentence():
    mats=[material('M1','中华人民共和国民法典','第五百八十五条'),material('M2','最高人民法院关于适用《中华人民共和国民法典》合同编通则若干问题的解释','第六十五条')]
    answer='民法典第五百八十五条[M2]和解释第六十五条[M1]。'
    assert article_grounding_issues(answer,mats)
    assert audit_citations(answer,mats,mats)['false_citation']


@pytest.mark.parametrize('name',['完全虚构解释','最高人民法院关于审理虚构纠纷案件的规定'])
def test_generic_suffix_does_not_accept_fabricated_name(name):
    mats=[material('M1','最高人民法院关于审理民间借贷案件适用法律若干问题的规定','第五条')]
    answer=name+'第五条[M1]。'
    assert article_grounding_issues(answer,mats)
    assert audit_citations(answer,mats,mats)['false_citation']


def test_independent_audit_rejects_bare_article_version_ambiguity():
    mats=[material('M1','中华人民共和国民法典','第五条','2020'),material('M2','中华人民共和国民法典','第五条','2026')]
    assert audit_citations('第五条[M1][M2]。',mats,mats)['false_citation']
