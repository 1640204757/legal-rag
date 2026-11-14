"""检索诊断尺子：看正确条文在关键词、语义两路和线上前4/前8里各排第几，定位丢在哪一步。

入口 main；只读连接本机 Qdrant/Ollama（复用 open_runtime，拒绝写库），不调用回答模型、不改检索代码。
dev 与 v2 输出逐题明细；holdout 和 --file 指定的外部题集只输出汇总，逐题内容不落盘。
设计与阅读顺序见 docs/guides/RETRIEVAL_MEASUREMENT.md。
"""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".local-deps"))

from local_rag.daily_eval_runtime import execution_sources, load_dataset, open_runtime, sha256_file
from local_rag.retrieval import DOC_FORMATS, Retriever, expand_colloquial_query
import local_rag.retrieval as retrieval_module
import local_rag.ollama as ollama_module
from local_rag.retrieval_measure import (
    ROUTE_DEPTH, build_key_index, compare_snapshots, gold_indices, make_record, normalize_external,
    public_records, snapshot, summarize,
)

V2_PATH = ROOT / "data/eval/retrieval_eval_v2.json"
SOURCES = ("scripts/measure_retrieval.py", "local_rag/retrieval_measure.py", "local_rag/ollama.py")
ROUTE_LABELS = {"bm25_raw": "关键词（原话）", "bm25": "关键词（扩展后）", "dense": "语义"}

# 四种查询任务说明的措辞在跑任何一次测量之前就写死在这里，不允许看结果再改。
# 法条入库不加前缀，所以四个变体共用同一批向量，不需要重建索引。
QUERY_PREFIX_VARIANTS = {
    "baseline": ollama_module.DEFAULT_QUERY_PREFIX,
    "v1": ("Instruct: Given a question written by an ordinary person in everyday spoken Chinese "
           "describing their own situation, retrieve the Chinese statute articles that govern that "
           "situation.\nQuery: "),
    "v2": "Instruct: 根据普通人用日常口语描述的亲身遭遇，检索规定这种情形的中国法律条文。\nQuery: ",
    "v3": ("Instruct: Given a question written by an ordinary person in everyday spoken Chinese "
           "describing their own situation, retrieve the Chinese statute articles that govern that "
           "situation. Focus on the legal relationship and the right or obligation in question, "
           "not on emotional words.\nQuery: "),
}



@contextmanager
def alias_ablation(enabled):
    """仅在本测量进程暂时关闭整张口语表，退出时原样恢复；不改文件、网页或索引。

    同时影响依赖该表的词面扩展及锚点，不是只删几个追加词的试验。
    进程级替换只供顺序诊断使用，不可嵌入有并发请求的生产服务。
    """
    original = retrieval_module.COLLOQUIAL_ALIASES
    try:
        if enabled:
            retrieval_module.COLLOQUIAL_ALIASES = {}
        yield
    finally:
        retrieval_module.COLLOQUIAL_ALIASES = original


async def production_search(retriever, query, top_k):
    """原样调用生产 search 取前 top_k 条；沿用日常评测口径，没有合格向量时记为降级模式。"""
    store = retriever.store
    if store:
        store.qualified_count = 0
    hits, mode = await retriever.search(query, top_k=top_k)
    if mode == "hybrid_qdrant" and not getattr(store, "qualified_count", 0):
        mode = "lexical_no_qualified_dense"
    return hits, mode


async def dense_ranking(retriever, query, depth, index_of):
    """不卡0.38地只读查询语义前 depth 名，返回 [(行号, 相似度)]；向量不可用时返回空列表。

    生产 QdrantStore.search 写死了阈值，看不到被挡掉的条文，所以这里另发同样范围的只读 query 请求。
    """
    store = retriever.store
    if not retriever.vectors or store is None or not store.ready:
        return []
    async with asyncio.timeout(45):
        vector = Retriever.normalize((await retriever.embedder.embed([query], query=True))[0])
    result = await store.request("POST", f"/collections/{store.collection}/points/query",
                                 {"query": vector, "filter": store.scope(), "limit": depth,
                                  "with_payload": ["chunk_id"]})
    return [(index_of[p["payload"]["chunk_id"]], p["score"]) for p in result["points"]
            if (p.get("payload") or {}).get("chunk_id") in index_of]


async def tail_ranking(retriever, query, exclude_ids, index_of):
    """按页面实际调用的 tail_candidates() 取"更多候选"并换成行号；失败时和页面一样按空尾部处理。"""
    method = getattr(retriever, "tail_candidates", None)
    if method is None:
        return []
    try:
        rows = await method(query, exclude_ids)
    except Exception:
        return []
    return [index_of[row["id"]] for row in rows if row.get("id") in index_of]


async def measure(retriever, questions, *, depth=ROUTE_DEPTH):
    """逐题收集各路排名并整理成测量记录；单题出错记为"出错"，不中断整轮、不移出分母。"""
    rows = retriever.rows
    key_index = build_key_index(rows)
    index_of = {row["id"]: i for i, row in enumerate(rows)}
    ids = [row["id"] for row in rows]
    records = []
    for question in questions:
        query = question["query"]
        expanded_query = expand_colloquial_query(query)
        try:
            # 生产 search() 只返回截断后的结果，看不到第21名以后；这里只读内存索引的完整排名，不修改它。
            bm25_raw = retriever._bm25.rank(query)[:depth]
            bm25 = retriever._bm25.rank(expanded_query)[:depth]
            dense = await dense_ranking(retriever, query, depth, index_of)
            # 前4、前8分别调用：复合问题的锚点重排会随 top_k 变化，不能用前8截出前4。
            calls = [await production_search(retriever, query, top_k) for top_k in (4, 8, depth)]
            (top4, mode), (top8, _), (pool, _) = [([index_of[h["id"]] for h in hits], m) for hits, m in calls]
            error = None if len({m for _, m in calls}) == 1 else "同一题几次检索的模式不一致"
            tail = await tail_ranking(retriever, query, {ids[i] for i in top8}, index_of)
        except Exception as exc:
            bm25_raw = bm25 = dense = top4 = top8 = pool = tail = []
            mode, error = "error", type(exc).__name__
        records.append(make_record(question, gold_indices(question, key_index), bm25_raw=bm25_raw, bm25=bm25,
                                   dense=dense, top4=top4, top8=top8, pool=pool, ids=ids, mode=mode,
                                   expanded=expanded_query != query, error=error, tail=tail))
    return records


def exit_code(records, *, source_changed=False):
    """出现降级、出错或运行中源码改变时返回2，避免把不正常的成绩当成混合检索结果。"""
    abnormal = any(r["mode"] != "hybrid_qdrant" or r["error"] for r in records)
    return 2 if abnormal or source_changed else 0


def write_run(output_root, summary, records, snap, *, stamp=None):
    """新建唯一运行目录，写汇总、dev/v2 明细和快照三个文件；不覆盖已有运行，返回该目录。

    目录按启动时间命名；Windows 时钟刷新间隔比微秒粗，同一时刻的两次运行会撞名，撞名时加序号。
    """
    stamp = stamp or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    root = Path(output_root)
    directory, n = root / stamp, 1
    while True:
        try:
            directory.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            n += 1
            directory = root / f"{stamp}-{n}"
    for name, value in (("summary.json", summary), ("details.json", public_records(records)),
                        ("snapshot.json", snap)):
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return directory


def load_questions(args, rows):
    """按参数取题：--file 只用外部题集；否则校验冻结的 v3 题集，再按 --split 取 dev(+holdout) 和 v2。"""
    if args.file:
        spec = json.loads(Path(args.file).read_text(encoding="utf-8"))
        return normalize_external(spec), dict(file_sha256=sha256_file(args.file))
    dataset, digest = load_dataset(args.dataset, rows)
    questions = [q for q in dataset["questions"] if args.split == "all" or q["split"] == "dev"]
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))
    # 与 eval_retrieval.py 相同：只取有明确期望条文的36道 v2 题，跳过范围降级题。
    questions += [dict(id="V2-" + q["id"], query=q["query"], split="v2", category=q["domain"],
                       expect_any=q["expect_hit_articles"]) for q in v2["questions"] if not q["expect_degrade"]]
    return questions, dict(dataset_sha256=digest, v2_sha256=sha256_file(V2_PATH))


def _show(value):
    """名次为空表示前60名内没有，打印成短横线。"""
    return "-" if value is None else value


def print_summary(splits):
    """把各组汇总打印成中文短表；只有数字，不打印 holdout 或外部题集的逐题内容。"""
    for split, s in splits.items():
        print(f"\n== {split}：{s['count']}题  前4命中 {s['top4_hits']}  前8命中 {s['top8_hits']}  "
              f"合并池命中 {s['pool_hits']}  前8+尾部命中 {s['display_hits']}")
        for route, label in ROUTE_LABELS.items():
            print(f"  {label}：" + "  ".join(f"前{d}={n}" for d, n in s["routes"][route].items()))
        print("  归类：" + "，".join(f"{c} {n}" for c, n in s["diagnosis"].items())
              + f"（排不上的里面，合并池外 {s['outside_pool']}）")
        a = s["alias_effect"]
        print(f"  对照表触发 {a['triggered']} 题：帮忙 {a['better']}，帮倒忙 {a['worse']}，没影响 {a['same']}")
        if s["errors"] or set(s["mode_counts"]) != {"hybrid_qdrant"}:
            print(f"  注意：检索模式 {s['mode_counts']}，出错 {s['errors']} 题")


def print_misses(records):
    """只为 dev、v2 打印没命中的题及各路名次，方便调试。"""
    for r in public_records(records):
        if r["diagnosis"] == "命中":
            continue
        score = "-" if r["dense_score"] is None else f"{r['dense_score']:.3f}"
        print(f"{r['id']} [{r['diagnosis']}] 原话{_show(r['bm25_raw_rank'])} 扩展{_show(r['bm25_rank'])} "
              f"语义{_show(r['dense_rank'])}(相似度{score}) 前8{_show(r['top8_rank'])} 池{_show(r['pool_rank'])}  "
              f"{r['query']}")


async def run(args):
    """连接只读运行环境，测量、留证并打印；返回退出码。"""
    started = time.perf_counter()
    sources = execution_sources(*SOURCES)
    async with open_runtime(api_url=args.api_url, qdrant_url=args.qdrant_url,
                            ollama_url=args.ollama_url, doc_format=args.doc_format,
                            collection=args.collection) as (retriever, _, info):
        # 只在本次测量进程内替换查询前缀；不写文件、不改生产默认值，也不影响法条向量。
        prefix = QUERY_PREFIX_VARIANTS[args.query_prefix]
        retriever.embedder.query_prefix = prefix
        prefix_sha256 = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        questions, scope = load_questions(args, retriever.rows)
        print("文档格式：", args.doc_format, "；查询前缀：", args.query_prefix, prefix_sha256[:16], flush=True)
        print("运行来源：", json.dumps({k: info[k] for k in ("collection", "corpus_id", "embedding_id",
                                                           "degradation_reason")}, ensure_ascii=False), flush=True)
        without_aliases = getattr(args, "without_colloquial", False)
        with alias_ablation(without_aliases):
            records = await measure(retriever, questions, depth=args.depth)
    source_changed = sources != execution_sources(*SOURCES)
    snap = snapshot(records)
    summary = dict(splits=summarize(records), depth=args.depth, source_changed=source_changed,
                   variant="without_colloquial_table" if without_aliases else "production",
                   doc_format=args.doc_format,
                   query_prefix=args.query_prefix, query_prefix_sha256=prefix_sha256,
                   degradation_reason=info["degradation_reason"],
                   scope=dict(scope, corpus_id=info["corpus_id"], embedding_id=info["embedding_id"],
                              collection=info["collection"], doc_format=args.doc_format, sources=sources),
                   elapsed_seconds=round(time.perf_counter() - started, 2))
    if args.compare:
        previous = json.loads((Path(args.compare) / "snapshot.json").read_text(encoding="utf-8"))
        summary["comparison"] = compare_snapshots(previous, snap)
    directory = write_run(args.output_root, summary, records, snap)
    print_summary(summary["splits"])
    print_misses(records)
    if "comparison" in summary:
        print("\n与上次快照对比：", json.dumps(summary["comparison"], ensure_ascii=False))
    print(f"\n耗时 {summary['elapsed_seconds']} 秒；结果目录：{directory}")
    return exit_code(records, source_changed=source_changed)


def main(argv=None):
    """解析命令参数；降级、出错或源码中途改变时仍写报告，但退出码为2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "all"), default="dev",
                        help="dev=80道dev+36道v2；all 再加40道holdout（holdout只出汇总）")
    parser.add_argument("--file", type=Path, help="外部题集（如 user10），只输出汇总")
    parser.add_argument("--compare", type=Path, help="上次的运行目录，用来对比线上前4/前8快照")
    parser.add_argument("--without-colloquial", action="store_true",
                        help="仅在独立测量进程关闭整张口语表及依赖锚点；不修改生产配置")
    parser.add_argument("--query-prefix", choices=tuple(QUERY_PREFIX_VARIANTS), default="baseline",
                        help="查询侧 Instruct 前缀措辞；baseline 为生产默认，其余为写死的实验变体")
    parser.add_argument("--doc-format", choices=DOC_FORMATS, default="plain",
                        help="法条向量化格式；plain 为生产默认，law_header_v1 为法名+条号实验格式")
    parser.add_argument("--collection", help="只读测量时的Qdrant集合名；默认沿用Web声明的生产集合")
    parser.add_argument("--depth", type=int, default=ROUTE_DEPTH, help="两路各看到第几名，默认60")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/eval/daily_eval_v3.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/local-rag/measure-runs")
    parser.add_argument("--api-url", default="http://127.0.0.1:18765")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:16333")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args(argv)
    if args.depth < 8:
        parser.error("--depth 至少为8，才能同时看前4和前8")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
