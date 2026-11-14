"""快速检索评测：读取本机Web集合/指纹，只读Qdrant，不调用4B回答模型。

入口main；v3先校验后评分，dev/holdout/v2分开留证，异常与降级均保留分母。
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".local-deps"))

from local_rag.daily_evaluation import acceptance_gate, aggregate_retrieval, score_retrieval
from local_rag.daily_eval_runtime import actual_search, execution_sources, load_dataset, make_scope, open_runtime, save_report, sha256_file


def choose_baseline(paths, current):
    """选择最早的完整、同来源真实混合基线；跳过开发子集与其他语料，不挑选高低成绩。"""
    for path in sorted(paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("label") != "baseline" or report.get("source_changed")
                or any(report.get("summary", {}).get(s, {}).get("count") != n for s, n in (("dev", 80), ("holdout", 40), ("v2", 36)))
                or any(report["scope"].get(k) != current["scope"].get(k) for k in ("dataset_sha256", "v2_sha256", "corpus_id", "embedding_id"))
                or len(report.get("results", [])) != 156
                or any(r.get("mode") != "hybrid_qdrant" or r.get("error") for r in report["results"])):
            continue
        return path
    return None


def evaluation_exit_code(report, comparison, *, split, label):
    """区分真实基线记录、开发运行和完整验收；缺基线/降级/源码变更/退步均不能冒充成功。"""
    degraded = any(r["mode"] != "hybrid_qdrant" or r.get("error") for r in report["results"])
    acceptance = report.get("acceptance")
    failed_acceptance = split == "all" and label != "baseline" and (acceptance is None or not acceptance["passed"])
    return 2 if degraded or report["source_changed"] or comparison["holdout_regressions"] or failed_acceptance else 0


async def evaluate(args):
    """顺序执行生产检索、记录真实Top-4并写唯一报告；不改标注、词表或数据库。"""
    started = time.perf_counter()
    sources = execution_sources("scripts/eval_retrieval.py")
    async with open_runtime(api_url=args.api_url, qdrant_url=args.qdrant_url, ollama_url=args.ollama_url,
                            lexical=args.lexical) as (retriever, model, info):
        dataset, digest = load_dataset(args.dataset, retriever.rows)
        info["source_hashes"] = sources
        questions = [q for q in dataset["questions"] if args.split == "all" or q["split"] == args.split]
        v2_path = ROOT / "data/eval/retrieval_eval_v2.json"
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        questions += [dict(id="V2-" + q["id"], query=q["query"], split="v2", category=q["domain"],
                           expect_any=q["expect_hit_articles"]) for q in v2["questions"] if not q["expect_degrade"]]
        results = []
        print("真实来源：", json.dumps(info, ensure_ascii=False), flush=True)
        for q in questions:
            try:
                hits, mode = await actual_search(retriever, q["query"])
                row = score_retrieval(q, hits, mode=mode)
            except Exception as exc:
                row = score_retrieval(q, [], mode="error", error=type(exc).__name__)
            results.append(row)
            if len(results) % 20 == 0:
                print(f"已评测 {len(results)}/{len(questions)}", flush=True)
        report = dict(kind="retrieval", label=args.label, scope=make_scope(info, digest, sha256_file(v2_path), results),
                      runtime=info, degradation_reason=info["degradation_reason"], results=results,
                      summary=aggregate_retrieval(results), elapsed_seconds=time.perf_counter()-started,
                      source_changed=sources != execution_sources("scripts/eval_retrieval.py"))
        baseline_path = args.baseline
        if baseline_path is None:
            baseline_path = choose_baseline(args.output_root.glob("*/retrieval.json"), report)
        report["baseline_report"] = str(baseline_path) if baseline_path else None
        report["acceptance"] = acceptance_gate(json.loads(baseline_path.read_text(encoding="utf-8")), report) if baseline_path else None
        directory = save_report(report, args.output_root)
        saved = json.loads((directory / "retrieval.json").read_text(encoding="utf-8"))
        for split, stats in report["summary"].items():
            print(f"{split}: {stats['hits']}/{stats['count']} hit@4={stats['hit_at_4']} MRR={stats['mrr']}")
            for category, item in stats["categories"].items():
                print(f"  {category}: {item['hits']}/{item['count']}")
        for row in results:
            if not row["hit"] and (args.show_misses == "all" or row["split"] != "holdout"):
                actual = [(r["title"], r["locator"].split()[0]) for r in row["top_4"]]
                expected = [(r["title"], r["article_no"]) for r in row["expect_any"]]
                print(f"MISS {row['id']} {row['split']} expected={expected} actual={actual}")
        print("变化：", json.dumps(saved["comparison"], ensure_ascii=False))
        print("基线验收：", json.dumps(report["acceptance"], ensure_ascii=False))
        print("REPORT=", directory)
        if args.split == "all" and args.label != "baseline" and report["acceptance"] is None:
            print("无法验收：缺少完整且同来源的baseline，请先显式建立基线；本轮结果已留证。")
        return evaluation_exit_code(report, saved["comparison"], split=args.split, label=args.label)


def main(argv=None):
    """解析命令参数；自动降级仍写报告但退出2，避免CI把词面成绩当混合验收成功。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/eval/daily_eval_v3.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/local-rag/eval-runs")
    parser.add_argument("--api-url", default="http://127.0.0.1:18765")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:16333")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--label", default="manual")
    parser.add_argument("--split", choices=("dev", "all"), default="dev", help="开发默认只运行dev；完整验收显式指定all")
    parser.add_argument("--baseline", type=Path, help="冻结基线报告；默认寻找同题集首个baseline运行")
    parser.add_argument("--lexical", action="store_true")
    parser.add_argument("--show-misses", choices=("all", "dev"), default="dev",
                        help="调参时用dev隐藏holdout错题细节；完整逐题结果仍保存供验收")
    return asyncio.run(evaluate(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
