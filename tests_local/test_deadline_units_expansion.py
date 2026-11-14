"""时间单位扩充回归：未知期限必须拦截，合法词面对照不新增误拦。"""
import pytest
from local_rag.service import deadline_grounding_issues


@pytest.mark.parametrize('text', ['期限通常为两周', '通知期限为两个星期', '至少半个月',
                                  '不得超过一个季度', '应在三周内提出', '期限一般为半个月',
                                  '应提前两周通知', '两个星期的通知期', '办理周期为一季度'])
def test_new_duration_without_evidence_is_blocked(text):
    assert deadline_grounding_issues(text + '，依据[M1]。', [dict(material_id='M1', text='应提供证明材料')])


@pytest.mark.parametrize('answer,source', [('期限为两周', '申请期限为两个星期'),
                                         ('至少半个月', '至少半个月'),
                                         ('期限为一个季度', '期限为一季度')])
def test_supported_new_duration_is_allowed(answer, source):
    assert not deadline_grounding_issues(answer + '，依据[M1]。', [dict(material_id='M1', text=source)])


def test_half_month_is_not_silently_converted_to_fifteen_days():
    assert deadline_grounding_issues('期限为半个月，依据[M1]。', [dict(material_id='M1', text='十五日内申请')])
