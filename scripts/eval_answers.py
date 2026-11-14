"""端到端抽检：默认固定抽20道dev题，调用真实本机4B及生产ChatService并逐题留证。

入口main；检查原始声明、展示编号、标题、步骤、跨场景机构、显式引用与650字篇幅。
这是结构自检，不是法律结论自动判卷；出错/截断保留，不重试挑分，不用holdout调提示词。
"""
import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".local-deps"))

import httpx

from local_rag.daily_evaluation import aggregate_answers, audit_answer, compare_answer_runs, sample_answers
from local_rag.daily_eval_runtime import actual_search, execution_sources, load_dataset, make_scope, open_runtime, prompt_materials, save_report
from local_rag.eval_runtime import ObservedModel, RecordingTransport, stream_answer
from local_rag.official_capture import fetch_official_npc_slice
from local_rag.service import ChatService


class ObservedRetriever:
    """透传生产检索并纠正评测模式，保留原始候选；不改变生产排序或问句。"""
    def __init__(self, inner):
        self.inner, self.rows = inner, inner.rows
        self.mode = "not_searched"

    async def search(self, query, top_k=4):
        """只记录本次真实合格向量模式；供服务继续完成相同预算/生成流程。"""
        hits, self.mode = await actual_search(self.inner, query, top_k=top_k)
        return hits, self.mode


async def evaluate(args):
    """逐题建立新内存会话，真实推理并即时写证据；结束释放连接，不操作Web用户会话。"""
    started = time.perf_counter()
    source_hashes = execution_sources("scripts/eval_answers.py")
    async with open_runtime(api_url=args.api_url, qdrant_url=args.qdrant_url, ollama_url=args.ollama_url) as (retriever, embedder, info):
        dataset, digest = load_dataset(args.dataset, retriever.rows)
        info["source_hashes"] = source_hashes
        questions = sample_answers(dataset["questions"], args.sample, seed=args.seed, split=args.split)
        if info["degradation_reason"] or info["model_digest"] == "missing" or ":4b" not in info["model"].lower():
            raise RuntimeError("真实4B/混合服务未就绪，拒绝用模拟或词面替代端到端抽检：" + info["degradation_reason"])
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        evidence = args.output_root / (stamp + "-answers-evidence")
        evidence.mkdir(parents=True, exist_ok=False)
        events = []
        model = ObservedModel(args.ollama_url, model=info["model"], embedding=embedder.embedding)
        await model.http.aclose()
        model.http = httpx.AsyncClient(base_url=model.base_url, trust_env=False, follow_redirects=False,
                                      timeout=httpx.Timeout(args.timeout, connect=5), transport=RecordingTransport(events, "ollama"))
        # 独立服务使用同一生产链路及默认Web官方抓取器，评测题/期望标签不进入检索或模型消息。
        observed = ObservedRetriever(retriever)
        service = ChatService(observed, model, timeout=args.timeout,
                              official_fetcher=None if args.offline_official else fetch_official_npc_slice)
        results = []
        try:
            for q in questions:
                question_started = time.perf_counter()
                events.clear()
                model.last_messages = []
                observed.mode = "not_searched"
                sid = service.create()
                rid = service.start(sid, q["query"])
                job = service.sessions[sid].jobs[rid]
                await job["task"]
                status = service.result(sid, rid)
                result = status.get("result", {})
                raw, reasons = stream_answer(events)
                raw = result.get("raw_answer", raw)
                prompt = model.last_messages[-1]["content"] if model.last_messages else ""
                # 正常结果中的retrieved就是预算后完整材料；异常时从真实输入逐字反查，不把Top4当已送模。
                materials = result.get("retrieved", prompt_materials(prompt))
                error = status.get("error") if status["status"] != "done" else None
                chat_calls = [e for e in events if e["url"].endswith("/api/chat")]
                if not error and (not reasons or any(reason == "length" for reason in reasons)
                                  or not chat_calls or not all(e.get("complete") and e.get("status_code") == 200 for e in chat_calls)):
                    error = "MODEL_STREAM_INCOMPLETE"
                checks = audit_answer(q, result, materials, error=error)
                row = dict(id=q["id"], query=q["query"], category=q["category"], split=q["split"],
                           expect_scenario=q["expect_scenario"], expect_any=q["expect_any"],
                           materials=materials, raw_answer=raw, answer=result.get("answer", ""),
                           result=result, checks=checks, model_stop_reasons=reasons,
                           mode=observed.mode, model_messages=model.last_messages, network_chat=chat_calls,
                           elapsed_seconds=time.perf_counter() - question_started)
                results.append(row)
                (evidence / (q["id"] + ".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                (evidence / "progress.json").write_text(json.dumps(dict(completed=len(results), total=len(questions), last_id=q["id"])), encoding="utf-8")
                service.sessions.pop(sid, None)
                print(f"{len(results)}/{len(questions)} {q['id']}: {json.dumps(checks, ensure_ascii=False)}", flush=True)
            if execution_sources("scripts/eval_answers.py") != source_hashes:
                raise RuntimeError("运行中源码变更，本轮不能作为同口径验收；逐题证据已保留")
            summary = aggregate_answers([r["checks"] for r in results])
            report = dict(kind="answers", scope=make_scope(info, digest, "", results), runtime=info,
                          evidence_directory=str(evidence), results=results, summary={args.split: summary},
                          elapsed_seconds=time.perf_counter()-started, degradation_reason=info["degradation_reason"],
                          selection=dict(seed=args.seed, sample=args.sample, split=args.split),
                          limitation="仅结构/显式引用检查，标签AI草拟待人工抽检；不证明事实、法律推理或隐含幻觉正确")
            previous = json.loads(args.baseline.read_text(encoding='utf-8')) if args.baseline else None
            report['answer_comparison'] = compare_answer_runs(previous, report)
            report['baseline_report'] = str(args.baseline) if args.baseline else None
            directory = save_report(report, args.output_root)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print("REPORT=", directory, "EVIDENCE=", evidence)
            return 2 if summary["errors"] or any(r["mode"] != "hybrid_qdrant" for r in results) or (args.baseline and not report['answer_comparison']['comparable']) else 0
        finally:
            await model.close()


def main(argv=None):
    """读取本机连接与抽样参数；默认只抽dev，holdout须显式指定且仅供泛化观察。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/eval/daily_eval_v3.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/local-rag/eval-runs")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument('--baseline', type=Path, help='同一固定样本的修复前回答报告，用相同评分规则分层对比')
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--api-url", default="http://127.0.0.1:18765")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:16333")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--offline-official", action="store_true", help="明确禁用官方在线补充，报告中仍保留缺口")
    return asyncio.run(evaluate(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
