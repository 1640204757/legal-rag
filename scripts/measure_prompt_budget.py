"""送模预算尺子：执行真实ChatService._run组装，在chat入口截获消息，绝不调用回答模型。

入口main；生产检索仍只读Qdrant/Ollama查询向量。官方切片仅复用已核验缓存/暂存，拒绝联网采集。
默认统计新会话；--history-file可注入合法历史做独立复测。holdout只保存分组汇总，不保存逐题。
"""
import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.local-deps'), str(ROOT)]

from local_rag.daily_eval_runtime import execution_sources, open_runtime, prompt_materials
from local_rag.evaluation import article_key
from local_rag.official_capture import fetch_official_npc_slice
from local_rag.service import ChatService
from scripts.measure_retrieval import load_questions


class AssemblyCaptured(Exception):
    """有意在回答模型调用前停止，不把预算统计伪装成已完成问答。"""


class CaptureModel:
    """只接收真实组装消息；不连接或调用4B模型。"""
    messages = None

    async def chat(self, messages):
        self.messages = [dict(m) for m in messages]
        raise AssemblyCaptured('预算观测已完成，未调用回答模型')


class CandidateRequest:
    """只在本测量器内改变主检索请求条数，记录预算前候选，不改变检索实现。"""
    def __init__(self, inner, top_k):
        self.inner, self.top_k, self.rows = inner, top_k, inner.rows

    async def search(self, query, top_k=4):
        self.hits, self.mode = await self.inner.search(query, top_k=self.top_k)
        return self.hits, self.mode


def reject_capture_client(*args, **kwargs):
    """在创建网络客户端前拒绝采集，防止只读统计写入官方缓存。"""
    raise RuntimeError('预算统计只读，禁止新建官方采集客户端')


async def cached_official_slice(entry, query):
    """沿用生产来源/哈希校验和选条，仅复用现有缓存或已验收暂存，无采集副作用。"""
    return await fetch_official_npc_slice(entry, query, client_factory=reject_capture_client)


class BudgetService(ChatService):
    """保留_run组装过程；省略不影响输入的页面展示查询，观测官方补充候选。"""
    supplements = ()

    async def _display_candidates(self, query):
        return [], []

    async def _official_supplements(self, *args):
        result = await super()._official_supplements(*args)
        self.supplements = result[0]
        return result


def target_present(question, materials):
    """仅评分实际送入材料，标签不参与检索或组装；返回至少一条可接受条文是否进入。"""
    targets = {(r['title'], article_key(r['article_no'])) for r in question['expect_any']}
    return any((m['title'], article_key(m['locator'])) in targets for m in materials)


async def probe_question(retriever, question, top_k, *, history=(), official_fetcher=cached_official_slice):
    """执行一题真实组装并返回数字记录；无回答推理、不写会话，不将候选当成已送模材料。"""
    request, model = CandidateRequest(retriever, top_k), CaptureModel()
    service = BudgetService(request, model, official_fetcher=official_fetcher, session_dir=False)
    sid = service.create()
    service.sessions[sid].history = [dict(m) for m in history]
    rid = service.start(sid, question['query'])
    await service.sessions[sid].jobs[rid]['task']
    main = list(getattr(request, 'hits', []))
    candidates = main + list(service.supplements)
    picked, count = [], 0
    for row in candidates:
        if count + len(row['text']) <= 3600:
            picked.append(row)
            count += len(row['text'])
    base = dict(top_k=top_k, mode=getattr(request, 'mode', 'not_searched'),
                main_candidates=len(main), supplements=len(service.supplements), candidates=len(candidates),
                char_budget_dropped=len(candidates)-len(picked))
    if model.messages is None:
        # 防御：生产装配已保证永不外抛超长异常；真出现空组装就记失败，不按“零丢弃”算成功。
        return dict(base, assembly_error=True, sent_count=None, target_sent=False)
    messages = model.messages
    materials = prompt_materials(messages[-1]['content'])
    # 确认取到的是完整条文；解析失败必须中止测量，不能默默低估材料数或字节数。
    actual = {(m['title'], m['locator'], m['version_id'], m['text']) for m in materials}
    offered = {(r['title'], r['locator'], r['version_id'], r['text']) for r in picked}
    if not actual <= offered or (not materials and '（本次无检索材料）' not in messages[-1]['content']):
        raise RuntimeError('不能从实际输入完整恢复材料，拒绝给出预算成绩')
    estimated = sum(len(m['content'].encode('utf-8')) + 32 for m in messages)
    if estimated > 7000:
        raise RuntimeError('实际组装超出生产硬预算')
    section = messages[-1]['content'].split('【检索材料】\n', 1)[1].split('\n【覆盖缺口】', 1)[0]
    return dict(base, assembly_error=False, sent_count=len(materials), target_sent=target_present(question, materials),
                byte_budget_dropped=len(picked)-len(materials), total_dropped=len(candidates)-len(materials),
                material_chars=sum(len(m['text']) for m in materials),
                material_bytes=sum(len(m['text'].encode('utf-8')) for m in materials),
                rendered_material_chars=len(section), rendered_material_bytes=len(section.encode('utf-8')),
                message_content_bytes=sum(len(m['content'].encode('utf-8')) for m in messages),
                message_estimated_bytes=estimated, system_bytes=len(messages[0]['content'].encode('utf-8')),
                # 只统计真正送进模型的历史视图：装配会把旧回答压成一行，注入字节不等于送模字节。
                history_messages_sent=len(messages) - 2,
                history_bytes=sum(len(m['content'].encode('utf-8')) for m in messages[1:-1]))


def aggregate(rows):
    """只聚合数字；holdout逐题记录不落盘。丢弃比例分母为实际返回候选总数。"""
    good = [r for r in rows if not r['assembly_error']]
    total = sum(r['candidates'] for r in good)
    metrics = ('sent_count', 'material_chars', 'material_bytes', 'rendered_material_chars', 'rendered_material_bytes',
               'message_content_bytes', 'message_estimated_bytes', 'system_bytes', 'history_bytes',
               'history_messages_sent')
    return dict(requested=len(rows), assembled=len(good), assembly_errors=len(rows)-len(good),
                mode_counts=dict(Counter(r['mode'] for r in rows)), target_sent=sum(r['target_sent'] for r in good),
                sent_count_histogram=dict(Counter(r['sent_count'] for r in good)),
                candidates=total, sent=sum(r['sent_count'] for r in good),
                char_budget_dropped=sum(r['char_budget_dropped'] for r in good),
                byte_budget_dropped=sum(r['byte_budget_dropped'] for r in good),
                total_dropped=sum(r['total_dropped'] for r in good),
                dropped_ratio=sum(r['total_dropped'] for r in good)/total if total else None,
                questions_with_drops=sum(r['total_dropped'] > 0 for r in good),
                post_notice_above_6800=sum(r['message_estimated_bytes'] > 6800 for r in good),
                metrics={key: dict(mean=statistics.mean(r[key] for r in good), maximum=max(r[key] for r in good))
                         for key in metrics} if good else {})


async def run(args):
    """读取冻结题集和可选历史，按三档观测并在项目内新建独立报告，不修改生产配置。"""
    source = execution_sources('scripts/measure_prompt_budget.py')
    history = json.loads(args.history_file.read_text(encoding='utf-8')) if args.history_file else []
    if (not isinstance(history, list) or len(history) % 2 or len(history) > 10
            or any(not isinstance(m, dict) or m.get('role') != ('user' if i % 2 == 0 else 'assistant')
                   or not isinstance(m.get('content'), str) for i, m in enumerate(history))):
        raise ValueError('历史必须是最多五轮、user/assistant交替的消息列表')
    if not args.output_root.resolve().is_relative_to(ROOT):
        raise ValueError('报告只能写入项目目录')
    async with open_runtime() as (retriever, _, info):
        if info['degradation_reason']:
            raise RuntimeError('真实混合环境未就绪')
        questions, scope = load_questions(argparse.Namespace(file=None, dataset=ROOT/'data/eval/daily_eval_v3.json', split='all'), retriever.rows)
        groups, details = defaultdict(list), []
        for serial, q in enumerate(questions):
            for top_k in (4, 5, 6):
                row = await probe_question(retriever, q, top_k, history=history)
                groups[top_k, q['split']].append(row)
                if q['split'] in {'dev', 'v2'}:
                    details.append(dict(row, id=q['id'], split=q['split']))
            if (serial+1) % 20 == 0:
                print(f'已组装 {serial+1}/{len(questions)}题（三档；未调用回答模型）', flush=True)
    changed = source != execution_sources('scripts/measure_prompt_budget.py')
    summary = dict(kind='prompt_budget', sources=source, source_changed=changed, scope=scope,
                   history_messages=len(history), official_policy='只复用已核验缓存/暂存；无缓存时保留目录提示，不联网采集',
                   limitation='默认新会话，不代表所有历史长度；只观测材料与预算，不衡量法律结论正确性。', depths={})
    for (top_k, split), rows in groups.items():
        summary['depths'].setdefault(str(top_k), {})[split] = aggregate(rows)
    stamp = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')
    directory, n = args.output_root / stamp, 1
    while True:
        try:
            directory.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            n += 1
            directory = args.output_root / f'{stamp}-{n}'
    for name, value in (('summary.json', summary), ('details.json', details)):
        (directory/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('REPORT=', directory)
    return 2 if changed or any(r['assembly_error'] or r['mode'] != 'hybrid_qdrant' for rows in groups.values() for r in rows) else 0


def main(argv=None):
    """命令行入口；全部产物写入项目内，输入历史只记录消息数/字节数，不复制正文。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-file',type=Path)
    parser.add_argument('--output-root',type=Path,default=ROOT/'artifacts/local-rag/prompt-budget-runs')
    return asyncio.run(run(parser.parse_args(argv)))


if __name__=='__main__':
    raise SystemExit(main())
