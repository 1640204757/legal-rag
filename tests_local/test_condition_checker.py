"""结构化明示条件三态逻辑测试；不把事实跨度存在当作法律语义认证。"""
from copy import deepcopy
import importlib.util
import unittest


class ConditionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('prototype.condition_checker'))
        from prototype import condition_checker
        self.api=condition_checker
        self.rule=dict(id='R01',fact_schema={
            'age':{'type':'integer','description':'年龄'},
            'notified':{'type':'boolean','description':'已告知'}},
            predicate={'all':[{'field':'age','op':'lt','value':14},{'field':'notified','op':'eq','value':True}]})
        self.case=dict(id='C01',rule_id='R01',context='年龄13岁，已经告知。',claim='卡上条件都满足',
                       assert_conditions_met=True,facts={
            'age':{'value':13,'evidence':{'start':2,'end':4,'text':'13'}},
            'notified':{'value':True,'evidence':{'start':6,'end':10,'text':'已经告知'}}})

    def test_supported_conditions_and_input_unchanged(self):
        before=deepcopy(self.case)
        result=self.api.check_conditions(self.rule,self.case)
        self.assertEqual(result['condition_state'],'MET')
        self.assertEqual(result['verdict'],'SUPPORT')
        self.assertEqual(self.case,before)
        self.assertFalse(result['legal_applicability_verified'])

    def test_explicit_false_is_not_missing(self):
        self.case['facts']['notified']['value']=False
        self.assertEqual(self.api.check_conditions(self.rule,self.case)['verdict'],'CONTRADICT')
        self.case['assert_conditions_met']=False
        self.assertEqual(self.api.check_conditions(self.rule,self.case)['verdict'],'SUPPORT')

    def test_missing_condition_stays_unknown(self):
        del self.case['facts']['notified']
        for assertion in (True,False):
            self.case['assert_conditions_met']=assertion
            self.assertEqual(self.api.check_conditions(self.rule,self.case)['verdict'],'UNKNOWN')

    def test_false_and_missing_means_all_conditions_not_met(self):
        self.case['facts']['age']['value']=15
        del self.case['facts']['notified']
        result=self.api.check_conditions(self.rule,self.case)
        self.assertEqual(result['condition_state'],'NOT_MET')
        self.assertEqual(result['missing_fields'],['notified'])

    def test_boolean_is_not_integer_and_no_string_coercion(self):
        for value in (True,'13',13.0,None):
            with self.subTest(value=value):
                case=deepcopy(self.case);case['facts']['age']['value']=value
                with self.assertRaises(ValueError):self.api.check_conditions(self.rule,case)

    def test_invalid_spans_and_missing_evidence_rejected(self):
        for evidence in (dict(start=0,end=2,text='13'),dict(start=True,end=4,text='13'),dict(start=2,end=4,text='14'),None):
            case=deepcopy(self.case);case['facts']['age']['evidence']=evidence
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                self.api.check_conditions(self.rule,case)

    def test_unknown_fields_and_rule_mismatch_rejected(self):
        self.case['facts']['invented']=deepcopy(self.case['facts']['age'])
        with self.assertRaises(ValueError):self.api.check_conditions(self.rule,self.case)
        del self.case['facts']['invented'];self.case['rule_id']='R99'
        with self.assertRaises(ValueError):self.api.check_conditions(self.rule,self.case)

    def test_numeric_boundary_operators(self):
        for op,value,expected in (('lt',13,False),('lte',13,True),('gt',13,False),('gte',13,True),('eq',13,True)):
            rule=deepcopy(self.rule);rule['predicate']['all'][0].update(op=op,value=value)
            with self.subTest(op=op):
                actual=self.api.check_conditions(rule,self.case)['condition_state']=='MET'
                self.assertEqual(actual,expected)

    def test_enum_values_strict_and_unknown_operator_rejected(self):
        rule=deepcopy(self.rule);rule['fact_schema']['age']={'type':'enum','values':['minor','adult'],'description':'类别'}
        rule['predicate']['all'][0].update(op='eq',value='minor')
        self.case['facts']['age']['value']='minor'
        self.assertEqual(self.api.check_conditions(rule,self.case)['verdict'],'SUPPORT')
        self.case['facts']['age']['value']='other'
        with self.assertRaises(ValueError):self.api.check_conditions(rule,self.case)
        rule=deepcopy(self.rule);rule['predicate']['all'][0]['op']='eval'
        with self.assertRaises(ValueError):self.api.check_conditions(rule,self.case)

    def test_empty_or_duplicate_predicates_rejected(self):
        for nodes in ([],[self.rule['predicate']['all'][0]]*2):
            rule=deepcopy(self.rule);rule['predicate']['all']=nodes
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                self.api.check_conditions(rule,self.case)

    def test_semantic_observation_cannot_override_condition_result(self):
        self.case['semantic_observations']=[{'verdict':'PASS'}]
        del self.case['facts']['age']
        result=self.api.check_conditions(self.rule,self.case)
        self.assertEqual(result['verdict'],'UNKNOWN')
        self.assertEqual(result['free_text_fact_extraction'],'NOT_IMPLEMENTED')

    def test_reviewed_integer_domain_rejects_negative_elapsed_days(self):
        self.rule['fact_schema']['age']['minimum']=0
        self.case['facts']['age']['value']=-1
        with self.assertRaises(ValueError):self.api.check_conditions(self.rule,self.case)

    def test_invalid_integer_domain_schema_rejected(self):
        for minimum in (True,'0'):
            rule=deepcopy(self.rule);rule['fact_schema']['age']['minimum']=minimum
            with self.subTest(minimum=minimum),self.assertRaises(ValueError):self.api.check_conditions(rule,self.case)


if __name__=='__main__':unittest.main()
