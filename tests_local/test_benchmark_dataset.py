"""挑战题集自洽性与失败门禁回归；使用真实本地语料，不访问模型或数据库。

负例只调用正常校验入口；不能在测试中附加特殊判断冒充校验器拦截。
"""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts.verify_benchmark_dataset import DATASET, ROOT, validate_dataset, verify_file
from local_rag.retrieval import load_corpus


@pytest.fixture(scope='module')
def inputs():
    return json.loads(DATASET.read_text(encoding='utf-8')), load_corpus()[0]


def test_frozen_real_dataset(inputs):
    data, rows = inputs
    assert validate_dataset(data, rows)['cases'] == 15
    assert verify_file(DATASET, rows)['cases'] == 15


@pytest.mark.parametrize('attack', [
    'missing_case','duplicate_id','wrong_category','fake_number','wrong_version','wrong_hash',
    'wrong_url','invented_fact','shift_offset','assistant_fact','duplicate_message',
    'overlapping_truth','unknown_field','bad_relation','singleton_alternative',
    'non_boolean','missing_clarification','missing_query_index','empty_core','empty_trap_reason',
    'wrong_trap_version','boolean_offset','wrong_corpus_hash',
])
def test_mutations_rejected(inputs, attack):
    data, rows = inputs
    data = copy.deepcopy(data)
    case = data['cases'][0]
    a = case['expected_core_articles'][0]
    f = case['extracted_facts'][0]
    if attack == 'missing_case': data['cases'].pop()
    elif attack == 'duplicate_id': data['cases'][1]['case_id'] = case['case_id']
    elif attack == 'wrong_category': case['category'] = 'unknown'
    elif attack == 'fake_number': a['article_number'] = '第九万九千九百九十九条'
    elif attack == 'wrong_version': a['version_id'] = 'NPC-20200528'
    elif attack == 'wrong_hash': a['text_sha256'] = '0'*64
    elif attack == 'wrong_url': a['source_url'] = 'https://example.com'
    elif attack == 'invented_fact': f['exact_text'] = '房东有全部过错'
    elif attack == 'shift_offset': f['start_char'] += 1
    elif attack == 'assistant_fact':
        c = data['cases'][10]
        text = c['dialogue_history'][1]['content']
        c['extracted_facts'][0] = dict(source_message_index=1, exact_text=text,start_char=0,end_char=len(text))
    elif attack == 'duplicate_message': data['cases'][10]['dialogue_history'][1]['message_index'] = 0
    elif attack == 'overlapping_truth':
        trap = {k:v for k,v in a.items() if k not in {'relation_type','alternative_group'}}
        case['forbidden_trap_articles'][0].update(trap)
    elif attack == 'unknown_field': case['hidden_answer'] = '绕过'
    elif attack == 'bad_relation': a['relation_type'] = 'required'
    elif attack == 'singleton_alternative': a.update(relation_type='alternative',alternative_group='g1')
    elif attack == 'non_boolean': case['clarification_needed'] = 'false'
    elif attack == 'missing_clarification': case['clarification_questions'] = []
    elif attack == 'missing_query_index': case.pop('query_message_index')
    elif attack == 'empty_core': case['expected_core_articles'] = []
    elif attack == 'empty_trap_reason': case['forbidden_trap_articles'][0]['trap_reason'] = ''
    elif attack == 'wrong_trap_version': case['forbidden_trap_articles'][0]['version_id'] = 'wrong'
    elif attack == 'boolean_offset': f['start_char'] = False
    elif attack == 'wrong_corpus_hash': data['corpus_rows_sha256'] = '0'*64
    with pytest.raises(ValueError):
        validate_dataset(data, rows)


def test_manifest_not_self_certified(inputs, tmp_path):
    data, rows = inputs
    data = copy.deepcopy(data)
    data['cases'][0]['query'] += '改题'
    target = tmp_path/'changed.json'
    target.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    with pytest.raises(ValueError, match='冻结'):
        verify_file(target, rows)


def test_cli_failure_nonzero(tmp_path):
    target = tmp_path/'bad.json'
    target.write_text('{}',encoding='utf-8')
    result = subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'scripts/verify_benchmark_dataset.py'),
                             '--dataset',str(target)],capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert result.returncode == 1
    assert 'FAIL' in result.stdout
