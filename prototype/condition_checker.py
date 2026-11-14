"""明示条件旁路：对已结构化事实做三值逻辑，不自动理解口语或法律适用。

主入口 check_conditions(rule, case)。规则只支持一层all及五种固定比较，无文件、
网络、模型或会话副作用。事实值与原话的语义对应须独立审查；跨度存在只防错绑。
"""
import operator
import re


OPERATORS={'eq':operator.eq,'lt':operator.lt,'lte':operator.le,'gt':operator.gt,'gte':operator.ge}


def typed_value(schema,value):
    """严格检查类型，布尔不冒充整数，字符串不自动转换成数值。"""
    kind=schema.get('type')
    if kind=='integer':
        return type(value) is int
    if kind=='boolean':
        return type(value) is bool
    if kind=='enum':
        return type(value) is str and value in schema['values']
    return False


def validate_rule(rule):
    """拒绝空规则、未知字段、动态表达式和不一致比较；不推断原文是否支持规则。"""
    if not isinstance(rule,dict) or not isinstance(rule.get('id'),str) or not rule['id']:
        raise ValueError('规则必须有ID')
    schema=rule.get('fact_schema')
    if not isinstance(schema,dict) or not 1<=len(schema)<=16:
        raise ValueError('规则字段必须为1到16项')
    for field,definition in schema.items():
        if not isinstance(field,str) or not re.fullmatch('[a-z][a-z0-9_]*',field) or not isinstance(definition,dict):
            raise ValueError('规则字段名称或定义无效')
        kind=definition.get('type')
        if kind not in ('boolean','integer','enum'):
            raise ValueError('未知字段类型')
        if 'minimum' in definition and (kind!='integer' or type(definition['minimum']) is not int):
            raise ValueError('minimum仅支持整数域的整数下界')
        if kind=='enum':
            values=definition.get('values')
            if not isinstance(values,list) or not values or any(type(x) is not str for x in values) or len(set(values))!=len(values):
                raise ValueError('enum必须提供不重复字符串列表')
    predicate=rule.get('predicate')
    if not isinstance(predicate,dict) or set(predicate)!={'all'}:
        raise ValueError('只允许一层all条件')
    nodes=predicate['all']
    if not isinstance(nodes,list) or not 1<=len(nodes)<=16:
        raise ValueError('all条件不能为空或超过16项')
    seen=set();used=set()
    for node in nodes:
        if not isinstance(node,dict) or set(node)!={'field','op','value'}:
            raise ValueError('条件字段不完整')
        field,op=node['field'],node['op']
        if not isinstance(field,str) or field not in schema or not isinstance(op,str) or op not in OPERATORS:
            raise ValueError('未知字段或比较操作')
        definition=schema[field]
        if not typed_value(definition,node['value']) or (op!='eq' and definition['type']!='integer'):
            raise ValueError('比较值类型不符')
        key=(field,op,node['value'])
        if key in seen:
            raise ValueError('重复条件')
        seen.add(key);used.add(field)
    if used!=set(schema):
        raise ValueError('字段定义与条件不对应')


def validate_case(rule,case):
    """核查结构化输入及原话跨度；不把字面存在认证为该事实值语义正确。"""
    if not isinstance(case,dict) or case.get('rule_id')!=rule['id']:
        raise ValueError('样本与规则绑定不符')
    context=case.get('context');claim=case.get('claim')
    if not isinstance(context,str) or not context.strip() or len(context)>16000 or not isinstance(claim,str) or not claim.strip():
        raise ValueError('事实原话或声明缺失/过长')
    if type(case.get('assert_conditions_met')) is not bool:
        raise ValueError('声明必须显式为布尔值')
    facts=case.get('facts')
    if not isinstance(facts,dict) or set(facts)-set(rule['fact_schema']):
        raise ValueError('事实字段未知')
    for field,fact in facts.items():
        if not isinstance(fact,dict) or not typed_value(rule['fact_schema'][field],fact.get('value')):
            raise ValueError('事实类型无效：'+field)
        definition=rule['fact_schema'][field]
        if 'minimum' in definition and fact['value']<definition['minimum']:
            raise ValueError('事实超出已登记输入域：'+field)
        evidence=fact.get('evidence')
        if not isinstance(evidence,dict):
            raise ValueError('事实缺少原话跨度')
        a,b,text=evidence.get('start'),evidence.get('end'),evidence.get('text')
        if type(a) is not int or type(b) is not int or not 0<=a<b<=len(context) or not isinstance(text,str) or not text.strip() or context[a:b]!=text:
            raise ValueError('事实原话跨度不符：'+field)


def check_conditions(rule,case):
    """返回本卡明示条件MET/NOT_MET/UNKNOWN及对结构化声明的三态意见。

    all中存在明确假即NOT_MET；没有假而缺事实才UNKNOWN。UNKNOWN既不变成假，
    也不能支持“条件不满足”的断言。结果仅旁路记录，不排序、删除或放行候选。
    """
    validate_rule(rule)
    validate_case(rule,case)
    facts=case['facts'];comparisons=[]
    for node in rule['predicate']['all']:
        field=node['field']
        if field not in facts:
            state='UNKNOWN'
        else:
            state='TRUE' if OPERATORS[node['op']](facts[field]['value'],node['value']) else 'FALSE'
        comparisons.append(dict(field=field,op=node['op'],expected_value=node['value'],state=state))
    states=[x['state'] for x in comparisons]
    condition_state='NOT_MET' if 'FALSE' in states else ('UNKNOWN' if 'UNKNOWN' in states else 'MET')
    if condition_state=='UNKNOWN':
        verdict='UNKNOWN'
    else:
        verdict='SUPPORT' if case['assert_conditions_met']==(condition_state=='MET') else 'CONTRADICT'
    return dict(case_id=case.get('id'),rule_id=rule['id'],condition_state=condition_state,verdict=verdict,
                comparisons=comparisons,missing_fields=sorted(set(rule['fact_schema'])-set(facts)),
                scope='STRUCTURED_CARD_CONDITIONS_ONLY',production_enabled=False,
                free_text_fact_extraction='NOT_IMPLEMENTED',fact_interpretation_verified=False,
                legal_applicability_verified=False,claim_text_interpretation='NOT_TESTED')
