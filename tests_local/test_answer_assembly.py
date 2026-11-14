"""结构化材料回答：程序切分/选号解析/确定性回填。

覆盖三类历史错法：
- D0406 类型：自由文本抄写漏掉句首前提；
- D0903 类型：列项括号/冒号改成了半角；
- D0101 类型：把提示说明一起抄进回答。

边界：完整旧阅读格式若已结构化但抄错，不自动改写，仍交给原校验器拦截；
恢复只用于“全部由材料行组成、但缺标题结构”的自由文本，避免把夹带结论的答案洗白。
"""
from local_rag.answer_assembly import (MATERIAL_SELECTION_SYSTEM, assemble_answer, build_final_answer,
                                       locate_complete_excerpt, split_source_excerpts)
from local_rag.answer_permission import is_complete_source_excerpt, material_reading_issues
from prototype.rag.input_builder import Material, RagInput


FOOD = ('生产不符合食品安全标准的食品或者经营明知是不符合食品安全标准的食品，消费者除要求赔偿损失外，'
        '还可以向生产者或者经营者要求支付价款十倍或者损失三倍的赔偿金；增加赔偿的金额不足一千元的，为一千元。'
        '但是，食品的标签、说明书存在不影响食品安全且不会对消费者造成误导的瑕疵的除外。')
PRIVACY = ('禁止在公共场所的下列区域、部位安装图像采集设备及相关设施：\n'
           '(一)旅馆、饭店、宾馆、招待所、民宿等经营接待食宿场所的客房或者包间内部；\n'
           '(二)学生宿舍的房间内部，或者单位为内部人员提供住宿、休息服务的房间内部；\n'
           '(三)公共的浴室、卫生间、更衣室、哺乳室、试衣间的内部。')


def material(material_id, text, title='测试规范'):
    return Material(material_id=material_id, kind='LAW_ARTICLE', title=title, locator='第一条',
                    version_id='测试', source_url='https://www.gov.cn/test', verified_scope='测试',
                    text=text, excerpts=split_source_excerpts(text))


def rows(materials):
    return [{'material_id': m.material_id, 'text': m.text} for m in materials]


def test_every_excerpt_passes_the_same_whole_sentence_validator():
    for text in (FOOD, PRIVACY, '第四十七条 甲。六个月以上不满一年的，按一年计算；不满六个月的，按半个月计算。'):
        excerpts = split_source_excerpts(text)
        assert excerpts and ''.join(excerpts) == text
        assert all(is_complete_source_excerpt(item, text) for item in excerpts)


def test_selection_is_filled_deterministically_from_material_text():
    materials = [material('M1', FOOD), material('M2', PRIVACY)]
    final, info = build_final_answer('SELECT: M1#1, M2#2\nASK: 价款十倍 | 旅馆、饭店、宾馆', materials)
    assert info['mode'] == 'selection' and info['selected'] == ['M1#1', 'M2#2']
    assert final.count('材料[M1]原文：“') == 1 and final.count('材料[M2]原文：“') == 1
    assert '材料[M1]中的“价款十倍”是否符合实际情况？' in final
    assert '材料[M2]中的“旅馆、饭店、宾馆”是否符合实际情况？' in final
    assert final.endswith('USED_IDS: M1 M2')
    assert not material_reading_issues(final, rows(materials))


def test_none_selection_uses_only_generic_questions_and_steps():
    materials = [material('M1', FOOD)]
    final, info = build_final_answer('SELECT: NONE\nASK: NONE', materials)
    assert info['mode'] == 'none'
    assert '本轮材料不足以直接说明所问情况。' in final
    assert '请补充事情发生的时间、参与主体和已有记录。' in final
    assert final.endswith('USED_IDS: 无')
    assert not material_reading_issues(final, rows(materials))


def test_unknown_excerpt_number_is_not_guessed_and_stays_blocked():
    materials = [material('M1', FOOD)]
    final, info = build_final_answer('SELECT: M1#99\nASK: NONE', materials)
    assert info['mode'] == 'invalid' and final.startswith('SELECT: M1#99')
    assert material_reading_issues(final, rows(materials))


def test_free_text_truncated_premise_recovers_the_whole_source_sentence():
    """D0406：自由文本只抄了后半句，程序按材料定位回完整句，不删前提。"""
    materials = [material('M1', FOOD)]
    raw = '材料[M1]原文：“' + FOOD[FOOD.index('消费者除'):] + '”'
    final, info = build_final_answer(raw, materials)
    assert info['mode'] == 'recovered'
    assert FOOD in final
    assert not material_reading_issues(final, rows(materials))


def test_free_text_half_width_punctuation_recovers_with_original_list_items():
    """D0903：模型把全角冒号改成半角，仍按原文回填，不截断列项边界。"""
    materials = [material('M1', PRIVACY)]
    prefix = PRIVACY[PRIVACY.index('禁止'):]
    truncated = prefix[:prefix.index('内部') + len('内部')]
    raw = '材料[M1]原文：“' + truncated.replace('\n', '').replace('：', ':') + '”'
    final, info = build_final_answer(raw, materials)
    assert info['mode'] == 'recovered'
    assert '材料[M1]原文：“禁止在公共场所的下列区域、部位安装图像采集设备及相关设施：' in final
    assert '(一)旅馆、饭店、宾馆' in final and '内部；”' in final
    assert not material_reading_issues(final, rows(materials))


def test_selection_drops_only_the_excerpt_with_unbindable_cross_reference():
    """第二十七条正文写着“本条例第八条”，现有条号校验无法绑定到本条 locator。

    这不是格式问题：改标点或加编号都会改动原文。装配只保留能通过原校验器的选段，
    被弃选段记入 dropped，校验器本身不放松。
    """
    cross = ('第二十七条 违反本条例第八条第一款规定安装图像采集设备及相关设施的，由公安机关没收相关设备设施。')
    materials = [material('M1', FOOD), material('M2', cross)]
    raw = 'SELECT: M1#1, M2#1\nASK: 价款十倍 | 没收相关设备设施'
    final, info = build_final_answer(raw, materials)
    assert info['mode'] == 'selection'
    assert info['selected'] == ['M1#1'] and info['dropped'] == ['M2#1']
    assert '材料[M1]原文：“' in final and '材料[M2]原文：“' not in final
    assert not material_reading_issues(final, rows(materials))
    from prototype.rag.grounding import article_grounding_issues
    assert not article_grounding_issues(final, rows(materials))


def test_selection_with_only_unbindable_cross_reference_stays_blocked():
    cross = ('第二十七条 违反本条例第八条第一款规定安装图像采集设备及相关设施的，由公安机关没收相关设备设施。')
    materials = [material('M1', cross)]
    final, info = build_final_answer('SELECT: M1#1\nASK: NONE', materials)
    assert info['mode'] == 'invalid' and final.startswith('SELECT: M1#1')


def test_structured_reading_with_truncated_quote_is_not_rewritten():
    """完整旧阅读格式抄错时维持原拦截，避免把错误抄写自动修补后当成通过。"""
    materials = [material('M1', FOOD)]
    raw = ('## 追问\n请补充事情发生的时间、参与主体和已有记录。\n'
           '## 处理方向\n材料[M1]原文：“消费者除要求赔偿损失外，还可以要求支付价款十倍。”\n'
           '1. 保存已有文件、凭证和沟通记录。\n2. 到官方文本核对版本、主体、条件及例外。\n'
           '## 证据清单\n保存与问题有关的原始文件、通知、凭证和沟通记录。\n'
           '## 不确定性\n以上是材料阅读线索，不认定个案适用；待核实事项并不穷尽。\nUSED_IDS: M1')
    final, info = build_final_answer(raw, materials)
    assert info['mode'] == 'structured_invalid' and final == raw
    assert material_reading_issues(final, rows(materials))


def test_extra_conclusion_is_not_laundered_by_quoting_a_valid_material():
    materials = [material('M1', FOOD)]
    raw = '材料[M1]原文：“' + FOOD + '”\n你肯定可以拿到十倍赔偿。'
    final, info = build_final_answer(raw, materials)
    assert info['mode'] == 'unparsed' and final == raw
    assert material_reading_issues(final, rows(materials))


def test_old_legal_reading_format_is_returned_byte_identically():
    materials = [material('M1', FOOD)]
    raw = ('## 追问\n请补充事情发生的时间、参与主体和已有记录。\n'
           '## 处理方向\n材料[M1]原文：“' + FOOD + '”\n'
           '1. 保存已有文件、凭证和沟通记录。\n2. 到官方文本核对版本、主体、条件及例外。\n'
           '## 证据清单\n保存与问题有关的原始文件、通知、凭证和沟通记录。\n'
           '## 不确定性\n以上是材料阅读线索，不认定个案适用；待核实事项并不穷尽。\nUSED_IDS: M1')
    assert build_final_answer(raw, materials) == (raw, {'mode': 'passthrough'})


def test_no_materials_returns_raw_without_touching_it():
    assert build_final_answer('任意原文', []) == ('任意原文', {'mode': 'no_materials'})


def test_selection_prompt_explains_program_fill_and_stays_short():
    assert '材料[M1]原文' in MATERIAL_SELECTION_SYSTEM
    assert len(MATERIAL_SELECTION_SYSTEM.encode('utf-8')) < 1200


def test_rendered_excerpt_markers_can_be_removed_to_restore_source_text():
    m = material('M1', PRIVACY)
    prompt = RagInput(question='问题', materials=[m], gaps=[]).render_user_prompt()
    assert '〔M1#1〕' in prompt and '〔M1#2〕' in prompt
    from local_rag.daily_eval_runtime import prompt_materials
    recovered = prompt_materials(prompt)
    assert recovered and recovered[0]['text'] == PRIVACY


def test_locate_complete_excerpt_never_returns_an_illegal_fragment():
    truncated = FOOD[FOOD.index('消费者除'):]
    recovered = locate_complete_excerpt(FOOD, truncated)
    assert recovered == FOOD and is_complete_source_excerpt(recovered, FOOD)
    assert locate_complete_excerpt(FOOD, '完全不存在的句子。') is None


# ---------------------------------------------------------------------------
# 次选异法专门条例过滤（2026-11-08）
# 现象：主选已命中通用大法，次选却拼进用户没提过的专门条例，
# 于是同一回答同时出现“六个月”和“十五日”这类互相冲突的期限。
# 判据：异法 + 次段法名带专门主题 + 问句没有该领域线索，三者同时满足才弃选。
# ---------------------------------------------------------------------------

ADMIN_LIMIT = '公民、法人或者其他组织直接向人民法院提起诉讼的，应当自知道或者应当知道作出行政行为之日起六个月内提出。'
VILLAGE_LIMIT = '当事人也可以自接到处罚决定通知之日起十五日内，直接向人民法院起诉。'
PENALTY_ARGUE = '行政机关必须充分听取当事人的意见，不得因当事人陈述、申辩而给予更重的处罚。'
TRAFFIC_ARGUE = '当事人对公安机关交通管理部门的处罚有权进行陈述和申辩，交通警察应当充分听取当事人的陈述和申辩。'
ASSAULT = '以暴力、胁迫或者其他方法强制猥亵他人或者侮辱妇女的，处五年以下有期徒刑或者拘役。'
RAILWAY_ORDER = '在列车内，寻衅滋事，扰乱公共秩序，危害旅客人身、财产安全的，铁路职工有权制止。'
LOAN_RULE = '出借人请求借款人按照合同约定利率支付利息的，人民法院应予支持。'
CONSUMER_TERM = '经营者不得以格式条款等方式作出排除或者限制消费者权利的规定。'
CONSUMER_REG = '经营者不得通过虚假或者引人误解的宣传，欺骗、误导消费者。'


def _check(final, materials):
    """弃选后仍必须过原有的原文与条号两个校验器，且不再展示被弃选的原文。"""
    from prototype.rag.grounding import article_grounding_issues
    assert not material_reading_issues(final, rows(materials))
    assert not article_grounding_issues(final, rows(materials))


def test_same_law_two_excerpts_are_both_kept():
    """同一部法的两条永远保留：二次引用不是“跨行业专门法”。"""
    materials = [material('M1', FOOD, title='中华人民共和国民法典'),
                 material('M2', LOAN_RULE, title='中华人民共和国民法典')]
    final, info = build_final_answer('SELECT: M1#1, M2#1\nASK: 价款十倍', materials,
                                     question='超市买到过期牛奶，能不能赔十份钱')
    assert info['selected'] == ['M1#1', 'M2#1'] and 'dropped' not in info
    assert '材料[M1]原文：“' in final and '材料[M2]原文：“' in final
    _check(final, materials)


def test_base_law_plus_judicial_interpretation_is_kept():
    """基础法 + 对应司法解释（民间借贷规定）不按专门条例处理，必须完整保留。"""
    materials = [material('M1', LOAN_RULE, title='最高人民法院关于审理民间借贷案件适用法律若干问题的规定'),
                 material('M2', FOOD, title='中华人民共和国民法典')]
    final, info = build_final_answer('SELECT: M1#1, M2#1\nASK: 支付利息', materials,
                                     question='网上借五千每天滚利息，这越滚越多受不了')
    assert info['selected'] == ['M1#1', 'M2#1'] and 'dropped' not in info
    _check(final, materials)


def test_consumer_law_and_its_implementing_regulation_are_kept():
    """消保法与其实施条例属于同一基础法族，不能当成跨行业专门法删掉。"""
    materials = [material('M1', CONSUMER_TERM, title='中华人民共和国消费者权益保护法'),
                 material('M2', CONSUMER_REG, title='中华人民共和国消费者权益保护法实施条例')]
    final, info = build_final_answer('SELECT: M1#1, M2#1\nASK: 格式条款', materials,
                                     question='饭馆门口写丢了东西概不负责，就能啥都不负责吗')
    assert info['selected'] == ['M1#1', 'M2#1'] and 'dropped' not in info
    _check(final, materials)


def test_special_law_is_kept_when_question_names_its_domain():
    """问句确实提到该专门领域（D0607 车祸→道路交通）时必须保留次选。"""
    materials = [material('M1', FOOD, title='中华人民共和国民法典'),
                 material('M2', TRAFFIC_ARGUE, title='中华人民共和国道路交通安全法实施条例')]
    final, info = build_final_answer('SELECT: M1#1, M2#1\nASK: 价款十倍', materials,
                                     question='出车祸后对方人跑了也不知道车谁的，医药钱先咋办')
    assert info['selected'] == ['M1#1', 'M2#1'] and 'dropped' not in info
    assert '材料[M2]原文：“' in final
    _check(final, materials)


def test_village_regulation_period_conflict_is_dropped():
    """D1209 类型：六个月内提出 + 村庄条例十五日内起诉，问句没有村庄/集镇线索，弃选次段。"""
    materials = [material('M1', ADMIN_LIMIT, title='中华人民共和国行政诉讼法'),
                 material('M2', VILLAGE_LIMIT, title='村庄和集镇规划建设管理条例')]
    final, info = build_final_answer('SELECT: M1#1, M2#1\nASK: 六个月内提出', materials,
                                     question='接到处罚四个月了家里才忙完，现在再去法院会不会太晚')
    assert info['selected'] == ['M1#1']
    assert info['dropped'] == ['M2#1'] and info['dropped_reasons'] == {'M2#1': 'period_conflict'}
    assert '六个月内提出' in final and '十五日内' not in final
    assert '材料[M2]原文：' not in final
    _check(final, materials)


def test_penalty_law_plus_traffic_regulation_drops_unrequested_special_law():
    """D1203 类型：行政处罚法 + 道路交通安全法实施条例，问句没有道路/车辆/交警线索。"""
    materials = [material('M1', PENALTY_ARGUE, title='中华人民共和国行政处罚法'),
                 material('M2', TRAFFIC_ARGUE, title='中华人民共和国道路交通安全法实施条例')]
    final, info = build_final_answer('SELECT: M1#1, M2#1\nASK: 陈述、申辩', materials,
                                     question='我想解释他们就说再顶嘴罚更多，这不是欺负人吗')
    assert info['selected'] == ['M1#1']
    assert info['dropped'] == ['M2#1']
    assert info['dropped_reasons'] == {'M2#1': 'unsolicited_special_law'}
    _check(final, materials)


def test_criminal_law_plus_railway_law_drops_unrequested_special_law():
    """D1102 类型：刑法 + 铁路法；问句说的是地铁猥亵，没提铁路/列车/火车。"""
    materials = [material('M1', ASSAULT, title='中华人民共和国刑法'),
                 material('M2', RAILWAY_ORDER, title='中华人民共和国铁路法')]
    final, info = build_final_answer('SELECT: M1#1, M2#1\nASK: 强制猥亵', materials,
                                     question='地铁上有人按着我乱摸我挣不开，我是男的也能报警管他吗')
    assert info['selected'] == ['M1#1']
    assert info['dropped'] == ['M2#1'] and '列车内' not in final
    _check(final, materials)


def test_without_question_the_filter_stays_off():
    """没有问句就不做取舍，保持改造前的逐字节行为。"""
    materials = [material('M1', ADMIN_LIMIT, title='中华人民共和国行政诉讼法'),
                 material('M2', VILLAGE_LIMIT, title='村庄和集镇规划建设管理条例')]
    final, info = build_final_answer('SELECT: M1#1, M2#1\nASK: NONE', materials)
    assert info['selected'] == ['M1#1', 'M2#1'] and 'dropped' not in info
    assert '十五日内' in final
