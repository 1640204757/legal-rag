"""离线证据旁路：核对原文身份与引用，用独立进程限制本地补证动作。

主入口 run_sidecar；check_evidence 是纯核验函数。读取传入的 JSON 对象，
不加载检索器或模型、不联网、不改候选顺序。进程临时结果写指定的隔离暂存目录。
目录由调用者提供，哈希绑定只防漂移，不认证目录来源或法律适用。
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import socket
import tempfile
import time


IDENTITY_FIELDS = ('id', 'text', 'source_url', 'version_id')
MAX_JSON_BYTES = 10 * 1024 * 1024


def canonical(value):
    """稳定序列化；拒绝 NaN 等不可复现的非标准 JSON 值。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def content_key(row):
    """同一来源、版本和正文换ID不构成新证据；不做语义近重复推断。"""
    return digest({key: row[key] for key in IDENTITY_FIELDS if key != 'id'})


def validate_input(packet, catalog):
    """限定输入规模和结构；有歧义的目录 ID 直接拒绝，不静默覆盖。"""
    if len(canonical([packet, catalog]).encode('utf-8')) > MAX_JSON_BYTES:
        raise ValueError('原型输入总量超过10MiB')
    if not isinstance(catalog, list) or len(catalog) > 10000:
        raise ValueError('catalog必须为不超过10000条的材料列表')
    index = {}
    for row in catalog:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in IDENTITY_FIELDS):
            raise ValueError('目录材料缺少非空身份字段')
        if row['id'] in index:
            raise ValueError('目录ID重复')
        index[row['id']] = row
    if not isinstance(packet, dict) or not isinstance(packet.get('question'), str) or not packet['question'].strip():
        raise ValueError('question必须为非空原问题')
    for key in ('candidates', 'citations', 'semantic_observations', 'actions'):
        if not isinstance(packet.get(key), list):
            raise ValueError(key + '必须显式提供列表')
    if len(packet['candidates']) > 128 or len(packet['actions']) > 16 or len(packet['citations']) > 128:
        raise ValueError('候选/引文/动作超过原型上限')
    return index


def check_evidence(packet, catalog):
    """纯核验：引文必须存在于绑定 ID 的已核对候选中；不推断语义。"""
    index = validate_input(packet, catalog)
    errors, verified, seen = [], {}, set()
    for row in packet['candidates']:
        key = row.get('id') if isinstance(row, dict) else None
        if not isinstance(key, str):
            errors.append({'kind': 'IDENTITY', 'reason': '候选缺少字符串ID'})
            continue
        reference = index.get(key)
        if key in seen or reference is None or any(row.get(k) != reference[k] for k in IDENTITY_FIELDS):
            errors.append({'kind': 'IDENTITY', 'id': key, 'reason': 'ID重复、未知或正文/来源/版本与目录不符'})
            # 同ID歧义时连先前出现的记录也不能继续提供引用背书。
            verified.pop(key, None)
        else:
            verified[key] = row
        seen.add(key)
    quote_errors = []
    for quote in packet['citations']:
        key = quote.get('id') if isinstance(quote, dict) else None
        text = quote.get('quote') if isinstance(quote, dict) else None
        if not isinstance(key, str) or key not in verified or not isinstance(text, str) or not text.strip() or text not in verified[key]['text']:
            quote_errors.append({'kind': 'CITATION', 'reason': '引文为空、ID未核验或不是该材料连续原文'})
    return dict(identity='FAIL' if errors else ('PASS' if packet['candidates'] else 'NOT_TESTED'),
                citations='FAIL' if quote_errors else ('PASS' if packet['citations'] else 'NOT_TESTED'),
                semantic='NOT_TESTED', version_applicability='NOT_TESTED', supplement_relevance='NOT_TESTED',
                errors=errors + quote_errors,
                identity_scope=list(IDENTITY_FIELDS))


def lookup_local(payload):
    """只按给定ID找同一目录；无检索、推断和文件/网络访问。"""
    return {'material': next((row for row in payload['catalog'] if row['id'] == payload['id']), None)}


def _verify(payload):
    return check_evidence(payload['packet'], payload['catalog'])


def _worker(function, payload, result_path):
    """可信内部动作的隔离执行入口；禁止socket连接，不支持任意用户代码。"""
    def denied(*args, **kwargs):
        raise RuntimeError('sidecar worker network denied')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    try:
        value = {'ok': True, 'value': function(payload)}
        encoded = canonical(value).encode('utf-8')
        if len(encoded) > MAX_JSON_BYTES:
            raise ValueError('工作结果超过原型上限')
    except Exception as error:
        encoded = canonical({'ok': False, 'error_type': type(error).__name__}).encode('utf-8')
    Path(result_path).write_bytes(encoded)


def _bounded_call(function, payload, deadline, scratch_dir):
    """同一绝对截止时间限制拥有的直接子进程；退出前不读取结果。

    不终止已有服务，也不声称取消远端任务或子孙进程。本原型白名单动作不派生子进程。
    超时后只做至多两秒清理；未确认退出时返回 STOP_CLEANUP_FAILED。
    """
    context = multiprocessing.get_context('spawn')
    with tempfile.TemporaryDirectory(prefix='rag-sidecar-', dir=scratch_dir) as directory:
        result_path = Path(directory) / 'result.json'
        process = context.Process(target=_worker, args=(function, payload, str(result_path)), daemon=True)
        process.start()
        pid = process.pid
        timed_out = False
        try:
            process.join(max(0, deadline - time.monotonic()))
            timed_out = time.monotonic() >= deadline
            if process.is_alive():
                timed_out = True
                process.terminate()
                process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join(1)
            exited = not process.is_alive()
            info = dict(pid=pid, worker_exited=exited, result_accepted=False,
                        exitcode=process.exitcode, termination_requested=timed_out)
            if not exited:
                return 'STOP_CLEANUP_FAILED', None, info
            if timed_out:
                return 'STOP_TIMEOUT', None, info
            if process.exitcode != 0 or not result_path.exists() or result_path.stat().st_size > MAX_JSON_BYTES:
                return 'STOP_WORKER_ERROR', None, info
            value = json.loads(result_path.read_text(encoding='utf-8'))
            if time.monotonic() >= deadline:
                return 'STOP_TIMEOUT', None, info
            if not value.get('ok'):
                info['error_type'] = value.get('error_type')
                return 'STOP_WORKER_ERROR', None, info
            info['result_accepted'] = True
            return None, value['value'], info
        finally:
            # 父请求被中断也只清理本次拥有的进程，防止后台迟到结果写回。
            if process.is_alive():
                process.kill()
                process.join(1)
            if not process.is_alive():
                process.close()


def validate_budget(max_calls, timeout_seconds):
    """预算预检无副作用，CLI在创建输出目录前调用。"""
    if type(max_calls) is not int or not 1 <= max_calls <= 16:
        raise ValueError('max_calls必须为1到16的整数')
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
        raise ValueError('timeout_seconds必须在(0,120]')


def run_sidecar(packet, catalog, *, max_calls=3, timeout_seconds=10, lookup=lookup_local, scratch_dir=None):
    """运行一次旁路核验，返回完整终态与trace；输入不变，补证永不插入原排序。

    lookup仅供开发者测试注入可信模块级函数；CLI不暴露动态函数或命令执行。
    新ID只表示结构性新增材料，相关性未认证，不能充当检索收益。
    """
    validate_budget(max_calls, timeout_seconds)
    started = time.monotonic()
    deadline = started + timeout_seconds
    index = validate_input(packet, catalog)
    # Windows受限环境的系统TEMP可能不允许子进程写入；使用明确的项目暂存位置。
    scratch_dir = Path(scratch_dir) if scratch_dir else Path(__file__).resolve().parents[1] / '.tmp/evidence-sidecar-workers'
    scratch_dir.mkdir(parents=True, exist_ok=True)
    baseline = deepcopy(packet)
    catalog = deepcopy(catalog)
    result = dict(status='RUNNING', baseline=baseline, baseline_sha256=digest(baseline),
                  checks=dict(identity='NOT_TESTED', citations='NOT_TESTED', semantic='NOT_TESTED',
                              version_applicability='NOT_TESTED', supplement_relevance='NOT_TESTED'),
                  semantic_observations={'verification': 'UNVERIFIED_INPUT', 'records': deepcopy(packet['semantic_observations'])},
                  supplemental=[], evidence_cards=[], gaps=['SEMANTICS_NOT_TESTED', 'VERSION_APPLICABILITY_NOT_TESTED'],
                  trace=[], calls_used=0, legal_correctness_verified=False,
                  budget=dict(max_calls=max_calls, timeout_seconds=timeout_seconds),
                  network_policy='NO_NETWORK_ACTIONS', model_requests=0)

    def finish(status):
        result['status'] = status
        if not baseline['candidates']:
            result['gaps'].append('NO_BASELINE_EVIDENCE')
        if not baseline['citations']:
            result['gaps'].append('NO_QUOTES_TO_VERIFY')
        if status.startswith('STOP_'):
            result['gaps'].append(status)
        result['elapsed_seconds'] = time.monotonic() - started
        result['baseline_unchanged'] = digest(result['baseline']) == result['baseline_sha256']
        return result

    def call(name, function, payload):
        if time.monotonic() >= deadline:
            return 'STOP_TIMEOUT', None
        if result['calls_used'] >= max_calls:
            return 'STOP_CALL_BUDGET', None
        result['calls_used'] += 1
        status, value, info = _bounded_call(function, payload, deadline, scratch_dir)
        result['trace'].append(dict(action=name, call_number=result['calls_used'], **info))
        return status, value

    status, checks = call('VERIFY_BASELINE', _verify, {'packet': baseline, 'catalog': catalog})
    if status:
        return finish(status)
    result['checks'] = checks
    if checks['identity'] == 'FAIL' or checks['citations'] == 'FAIL':
        return finish('STOP_INVALID_EVIDENCE')
    result['evidence_cards'] = [dict(id=row['id'], text=row['text'], source_url=row['source_url'],
                                   version_id=row['version_id'], label='程序附列目录原文，适用性未核验')
                                for row in baseline['candidates']]
    seen_actions = set()
    evidence_ids = {row['id'] for row in baseline['candidates']}
    evidence_content = {content_key(row) for row in baseline['candidates']}
    for action in baseline['actions']:
        if not isinstance(action, dict) or action.get('kind') != 'LOOKUP_LOCAL' or not isinstance(action.get('id'), str) or set(action) != {'kind', 'id'}:
            return finish('STOP_ACTION_NOT_ALLOWED')
        fingerprint = digest(action)
        if fingerprint in seen_actions:
            return finish('STOP_DUPLICATE_ACTION')
        seen_actions.add(fingerprint)
        state_hash = digest({'question': baseline['question'], 'evidence': sorted(evidence_ids),
                             'content': sorted(evidence_content)})
        status, found = call('LOOKUP_LOCAL', lookup, {'id': action['id'], 'catalog': catalog})
        if status:
            return finish(status)
        result['trace'][-1].update(action_sha256=fingerprint, state_sha256=state_hash)
        row = found.get('material') if isinstance(found, dict) else None
        if row is None or action['id'] in evidence_ids:
            result['trace'][-1]['new_material'] = False
            return finish('STOP_NO_PROGRESS')
        if not isinstance(row, dict) or row.get('id') != action['id'] or row != index.get(action['id']):
            return finish('STOP_INVALID_SUPPLEMENT')
        if content_key(row) in evidence_content:
            result['trace'][-1]['new_material'] = False
            return finish('STOP_NO_PROGRESS')
        evidence_ids.add(row['id'])
        evidence_content.add(content_key(row))
        result['supplemental'].append(row)
        result['trace'][-1].update(new_material=True, semantic_progress='NOT_TESTED')
    return finish('COMPLETE_WITH_UNVERIFIED_SEMANTICS')
