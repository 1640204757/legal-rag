"""评测口径回归：防止空分母虚高、同号串法、版本混用及假编号漏记。"""
import hashlib
import json
import pytest
from local_rag.evaluation import score_hits, audit_citations, judge_degrade, aggregate, verify_frozen


def question():
    return dict(expect_hit_docs=['测试法'],expect_hit_articles=[dict(title='测试法',version_id='v1',article_no='第十七条')],expect_degrade=False)


def row(title='测试法',version='v1',locator='第十七条 原文行L1'):
    return dict(title=title,version_id=version,locator=locator,text='原文',material_id='M1')


def test_exact_article_rejects_other_law_and_other_version():
    assert score_hits(question(),[row()])['article_all']
    assert not score_hits(question(),[row(title='其他法')])['article_any']
    assert not score_hits(question(),[row(version='v2')])['article_any']
    assert not score_hits(question(),[row(locator='第十七条之一')])['article_any']


def test_any_and_all_are_different():
    q=question()
    q['expect_hit_articles'].append(dict(title='测试法',version_id='v1',article_no='第十八条'))
    score=score_hits(q,[row()])
    assert score['article_any'] and not score['article_all']


def test_boundary_does_not_get_free_hit():
    score=score_hits(dict(expect_hit_docs=[],expect_hit_articles=[],expect_degrade=True),[])
    assert score['doc_hit'] is None and score['article_any'] is None


def test_unknown_declared_id_counts_even_without_body_reference():
    audit=audit_citations('资料如下。\nUSED_IDS: M99',[row()],[row()])
    assert audit['false_citation']


def test_nonexistent_article_detected_despite_valid_material_id():
    audit=audit_citations('依据《测试法》第九百九十九条办理。[M1]\nUSED_IDS: M1',[row()],[row()])
    assert audit['false_citation']


def test_wrong_law_same_number_is_not_supported():
    audit=audit_citations('根据《其他法》第十七条办理。[M1]\nUSED_IDS: M1',[row()],[row(),row('其他法')])
    assert audit['false_citation']


def test_valid_material_citation_and_article_pass():
    assert not audit_citations('依据《测试法》第十七条办理。[M1]\nUSED_IDS: M1',[row()],[row()])['false_citation']


def test_unbound_article_is_not_silently_passed():
    audit=audit_citations('根据第十七条可以申请。',[row()],[row()])
    assert audit['unverifiable']


def test_block_message_is_not_coverage_degrade():
    assert not judge_degrade('生成包含无法核对的引用，已拦截。',True,['加州'])
    assert judge_degrade('未收录加州当地法律，无法确认通知期限。',False,['加州'])
    assert not judge_degrade('未收录加州当地法律，但你必须在30日内起诉。',False,['加州'])


def test_aggregate_excludes_boundary_and_keeps_failed_execution():
    rows=[dict(id='Q1',expect_degrade=False,complete=True,retrieval=score_hits(question(),[row()]),assembly=score_hits(question(),[]),
               raw_audit=dict(false_citation=False,unverifiable=False),final_audit=dict(false_citation=False,unverifiable=False),degrade_correct=None),
          dict(id='Q2',expect_degrade=True,complete=False,retrieval=score_hits(dict(expect_hit_docs=[],expect_hit_articles=[]),[]),
               assembly=score_hits(dict(expect_hit_docs=[],expect_hit_articles=[]),[]),raw_audit=None,final_audit=None,degrade_correct=False)]
    report=aggregate(rows,[],expected_total=2)
    assert report['retrieval']['doc_hit_at4']['denominator']==1
    assert report['retrieval']['doc_hit_at4']['rate']==1
    assert report['execution']['completed']==1
    assert report['verdict']=='不达标'


def test_freeze_rejects_modified_spec(tmp_path):
    spec=tmp_path/'spec.json'
    spec.write_bytes(b'{}')
    manifest=tmp_path/'spec.frozen.json'
    manifest.write_text(json.dumps(dict(sha256=hashlib.sha256(b'{}').hexdigest())))
    verify_frozen(spec,manifest)
    spec.write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError):
        verify_frozen(spec,manifest)


@pytest.mark.parametrize('answer', [
    '未收录加州法律，但通知期限为30天。',
    '未收录南山区实施细则，申请条件是本区户籍。',
])
def test_coverage_disclaimer_cannot_hide_specific_invented_rule(answer):
    assert not judge_degrade(answer,False,['加州','南山'])


def test_valid_id_bound_to_wrong_article_is_false_citation():
    materials=[row(),dict(row('其他法',locator='第十八条'),material_id='M2')]
    audit=audit_citations('依据《测试法》第十七条办理。[M2]',materials,materials)
    assert audit['false_citation']


def test_unknown_law_says_provides_without_according_to_is_detected():
    assert audit_citations('《不存在法》规定可以申请。[M1]',[row()],[row()])['false_citation']
