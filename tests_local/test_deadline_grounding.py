"""维权期限真实性校对器（Deadline Grounding）自动化测试

用途说明：
验证 local_rag/service.py 中的 deadline_grounding_issues 函数。
法律问答中，大模型极易根据常识脑补“1年内申请”、“30日内起诉”等时限。
该校对器负责核验：模型回答里出现的每一个时间数字，是否在所引用的条文原文中真实出现。

测试覆盖点：
1. 法条里没写期限，模型擅自写了期限，必须判定违规并拦截。
2. 支持阿拉伯数字与中文大写数字的统一归一化比对（例如条文写“一年内”，模型写“1年内”，视为匹配通过）。
3. 跨法条混淆检测：期限出自材料 M2，但回答却标注依据 M1，必须拦截。
"""
from local_rag.service import deadline_grounding_issues


def test_existing_id_with_no_matching_duration_is_rejected():
    """测试时限无依据拦截：法条 M1 真实存在，但原文没有时限，模型擅自得出“1年内”时必须拦截"""
    answer = '职工自事故发生之日起1年内申请，依据M1。'
    materials = [{'material_id': 'M1', 'text': '申请应当提交诊断证明。'}]
    # 期望返回 DEADLINE_VALUE_NOT_IN_CITED_TEXT 警告列表（非空）
    assert deadline_grounding_issues(answer, materials)


def test_arabic_and_chinese_duration_match_is_only_lexical():
    """测试中阿数字词面匹配：原文是中文“一年”，模型生成阿拉伯“1年”，能够正确对照识别通过"""
    answer = '职工自事故发生之日起1年内申请，依据M1。'
    materials = [{'material_id': 'M1', 'text': '职工自事故发生之日起一年内提出申请。'}]
    # 期望没有拦截警告（返回空列表）
    assert not deadline_grounding_issues(answer, materials)


def test_duration_from_different_material_does_not_validate_wrong_id():
    """测试张冠李戴拦截：期限确实在材料 M2 中出现，但回答句子结尾却错误标注了依据 M1，判定违规拦截"""
    materials = [
        {'material_id': 'M1', 'text': '申请表'},
        {'material_id': 'M2', 'text': '一年内申请'}
    ]
    # 该句引用了 M1，但在 M1 找不到 1 年，判定拦截
    assert deadline_grounding_issues('职工在1年内申请，依据M1。', materials)

