"""检索诊断尺子的纯计算：给每道题找各路排名、归类，再分组汇总和对比快照。

阅读入口 build_key_index → gold_indices → make_record（内含 diagnose）→ summarize → compare_snapshots。
不联网、不调模型、不写盘；运行编排和留证在 scripts/measure_retrieval.py。
归类只说明正确条文丢在检索的哪一步，不判断条文是否适用于个案。
"""
from collections import Counter, defaultdict

from .evaluation import article_key

# 各路看到第60名，覆盖"第13–60名尾部候选"的范围。
ROUTE_DEPTH = 60
# 与 retrieval.CANDIDATE_LIMIT 一致：每路前20名才进合并池。
CANDIDATE_DEPTH = 20
# 与生产语义阈值一致（qdrant.py 的 score_threshold、retrieval.py 的 s >= 0.38），这里只用于归类。
DENSE_THRESHOLD = 0.38
DEPTHS = (4, 8, 20, 60)
# 允许逐题查看和保存的分组；holdout 与外部题集只出汇总。
DETAIL_SPLITS = frozenset({"dev", "v2"})
CATEGORIES = ("命中", "被0.38挡掉", "排不上", "找不到", "出错")
ROUTES = ("bm25_raw", "bm25", "dense")


def row_key(title, locator):
    """把法名和条号合成比较键；条号统一成数字，"第三条"和"第3条"视为同一条。"""
    return title, article_key(locator or "")


def build_key_index(rows):
    """一次性建立"法名+条号 → 行号列表"的索引，避免每道题都扫描全库；不修改输入。"""
    index = defaultdict(list)
    for i, row in enumerate(rows):
        index[row_key(row.get("title"), row.get("locator"))].append(i)
    return index


def gold_indices(question, key_index):
    """返回题目标注的全部可接受条文所在的行号；标注在库里找不到时返回空集合。"""
    return {i for target in question["expect_any"]
            for i in key_index.get(row_key(target["title"], target["article_no"]), ())}


def first_rank(ranking, gold, depth=ROUTE_DEPTH):
    """在前 depth 名里找第一个正确条文，返回从1开始的名次；depth 为 None 时看全部，没找到返回 None。"""
    for rank, index in enumerate(ranking[:depth], 1):
        if index in gold:
            return rank
    return None


def diagnose(record):
    """按"命中 → 被0.38挡掉 → 排不上 → 找不到"的先后归类；检索出错单列，仍留在分母里。"""
    if record["error"]:
        return "出错"
    if record["top4_rank"]:
        return "命中"
    # 语义其实排进了前20，却因相似度低于阈值没能进合并池：问题出在阈值，不在召回。
    if (record["pool_rank"] is None and record["dense_rank"] and record["dense_rank"] <= CANDIDATE_DEPTH
            and record["dense_score"] < DENSE_THRESHOLD):
        return "被0.38挡掉"
    # 原话 BM25 不是线上实际的一路，只用来对比对照表效果，不参与归类。
    if record["pool_rank"] or record["bm25_rank"] or record["dense_rank"]:
        return "排不上"
    return "找不到"


def make_record(question, gold, *, bm25_raw, bm25, dense, top4, top8, pool, ids, mode, expanded, error=None, tail=()):
    """把一道题各路结果整理成一条测量记录并归类，返回新字典。

    bm25_raw/bm25/dense 为 [(行号, 分数)]；top4/top8/pool 是线上 search 返回的行号次序；
    tail 是页面"更多候选"的行号次序；ids 把行号换成条文编号，供快照比较。
    """
    dense_rank = first_rank([i for i, _ in dense], gold)
    record = dict(
        id=question["id"], split=question["split"], category=question.get("category", ""),
        query=question["query"], expect_any=question["expect_any"], mode=mode, error=error, expanded=expanded,
        bm25_raw_rank=first_rank([i for i, _ in bm25_raw], gold), bm25_rank=first_rank([i for i, _ in bm25], gold),
        dense_rank=dense_rank, dense_score=dense[dense_rank - 1][1] if dense_rank else None,
        top4_rank=first_rank(top4, gold, 4), top8_rank=first_rank(top8, gold, 8), pool_rank=first_rank(pool, gold),
        top4=[ids[i] for i in top4[:4]], top8=[ids[i] for i in top8[:8]])
    record["tail_rank"] = first_rank(list(tail), gold, depth=None)
    # 页面能展示出正确条文：在前8条里，或在"更多候选"里。
    record["display_hit"] = record["top8_rank"] is not None or record["tail_rank"] is not None
    record["diagnosis"] = diagnose(record)
    # 只在某一路第21–60名出现、没进合并池：这部分正是"尾部候选"能补回来的。
    record["outside_pool"] = record["diagnosis"] == "排不上" and record["pool_rank"] is None
    return record


def _stats(items):
    """计算一组题的汇总数字；比率在没有题时为 None，不伪造0%或100%。"""
    count = len(items)

    def rate(n):
        return n / count if count else None

    def order(rank):
        # 前60名以外记作第61名，才能比较"扩展后"与"原话"谁更靠前。
        return rank if rank is not None else ROUTE_DEPTH + 1

    hits = {name: sum(r[f"{name}_rank"] is not None for r in items) for name in ("top4", "top8", "pool")}
    routes = {route: {depth: sum(r[f"{route}_rank"] is not None and r[f"{route}_rank"] <= depth for r in items)
                      for depth in DEPTHS} for route in ROUTES}
    triggered = [r for r in items if r["expanded"]]
    alias = dict(triggered=len(triggered),
                 better=sum(order(r["bm25_rank"]) < order(r["bm25_raw_rank"]) for r in triggered),
                 worse=sum(order(r["bm25_rank"]) > order(r["bm25_raw_rank"]) for r in triggered),
                 same=sum(order(r["bm25_rank"]) == order(r["bm25_raw_rank"]) for r in triggered))
    categories = {}
    for category in sorted({r["category"] for r in items}):
        group = [r for r in items if r["category"] == category]
        categories[category] = dict(count=len(group), top4_hits=sum(r["top4_rank"] is not None for r in group))
    return dict(count=count, errors=sum(bool(r["error"]) for r in items),
                mode_counts=dict(Counter(r["mode"] for r in items)),
                top4_hits=hits["top4"], top4_rate=rate(hits["top4"]),
                top8_hits=hits["top8"], top8_rate=rate(hits["top8"]),
                pool_hits=hits["pool"], pool_rate=rate(hits["pool"]),
                display_hits=sum(r["display_hit"] for r in items),
                display_rate=rate(sum(r["display_hit"] for r in items)),
                routes=routes, diagnosis={c: sum(r["diagnosis"] == c for r in items) for c in CATEGORIES},
                outside_pool=sum(r["outside_pool"] for r in items), alias_effect=alias, categories=categories)


def summarize(records):
    """按分组汇总命中、各路前4/8/20/60命中、归类和对照表效果；只有数字，不含逐题内容。"""
    groups = defaultdict(list)
    for r in records:
        groups[r["split"]].append(r)
    return {split: _stats(items) for split, items in groups.items()}


def snapshot(records):
    """只保留题号、分组和线上前4/前8的条文编号，供下次对比；不含题目原文和标准答案。"""
    return {r["id"]: dict(split=r["split"], top4=list(r["top4"]), top8=list(r["top8"])) for r in records}


def compare_snapshots(previous, current):
    """逐组对比两次快照的线上前4/前8；dev、v2 列出变化题号，其他组只报变化数量。"""
    result = {}
    for split in sorted({v["split"] for v in previous.values()} | {v["split"] for v in current.values()}):
        before = {q for q, v in previous.items() if v["split"] == split}
        after = {q for q, v in current.items() if v["split"] == split}
        if before != after:
            result[split] = dict(comparable=False, reason="两次的题号集合不同")
            continue
        changed = {k: sorted(q for q in after if previous[q][k] != current[q][k]) for k in ("top4", "top8")}
        entry = dict(comparable=True, top4_changed=len(changed["top4"]), top8_changed=len(changed["top8"]))
        if split in DETAIL_SPLITS:
            entry.update(top4_changed_ids=changed["top4"], top8_changed_ids=changed["top8"])
        result[split] = entry
    return result


def public_records(records):
    """只返回允许逐题保存的 dev、v2 记录；holdout 和外部题集的逐题内容不落盘。"""
    return [r for r in records if r["split"] in DETAIL_SPLITS]


def normalize_external(spec):
    """把外部题集（如 user10）整理成统一格式，分组记为 external；格式错误只报题序和字段名。

    接受 {"questions": [...]} 或直接的列表，标注字段可以叫 expect_any 或 expect_hit_articles。
    报错信息不回显题目原文和标注，避免留出题内容被打印出来。
    """
    items = spec.get("questions") if isinstance(spec, dict) else spec
    if not isinstance(items, list) or not items:
        raise ValueError("外部题集需要非空的 questions 列表")
    questions, errors = [], []
    for n, item in enumerate(items, 1):
        item = item if isinstance(item, dict) else {}
        targets = item.get("expect_any") or item.get("expect_hit_articles")
        missing = [field for field in ("id", "query") if not isinstance(item.get(field), str) or not item[field].strip()]
        if (not isinstance(targets, list) or not targets
                or not all(isinstance(t, dict) and t.get("title") and t.get("article_no") for t in targets)):
            missing.append("expect_any")
        if missing:
            errors.append(f"第{n}题缺少或无效：{', '.join(missing)}")
            continue
        questions.append(dict(id=item["id"], query=item["query"], split="external",
                              category=item.get("category") if isinstance(item.get("category"), str) else "外部题集",
                              expect_any=[dict(title=t["title"], article_no=t["article_no"]) for t in targets]))
    if errors:
        raise ValueError("外部题集格式不符：" + "；".join(errors))
    return questions
