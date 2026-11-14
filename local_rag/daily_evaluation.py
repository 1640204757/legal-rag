"""日常评测的纯评分口径：先校验标注，再计算命中、结构缺陷和可比运行变化。

阅读入口 validate_questions → score_retrieval/aggregate_retrieval → compare_runs。
本模块不联网、不调模型、不写盘；关键词校验仅证明文本支撑线索存在，不能替代人工法律抽检。
"""
from collections import Counter, defaultdict
import random
import re

from prototype.rag.citation_check import find_citations, parse_used_ids
from .evaluation import ARTICLE, article_key, audit_citations

DRAFT_STATUS = "AI草拟待人工抽检"
HEADINGS = ("追问", "处理方向", "证据清单", "不确定性")
INSTITUTIONS = ("单位法务", "教务处", "学生申诉", "员工手册", "单位管理责任")
ANSWER_FLAGS = ("blocked", "missing_used_ids", "display_internal_ids", "missing_headings", "missing_steps",
                "wrong_institutions", "bad_citations", "over_650", "raw_wrong_institutions",
                "raw_wrong_criminal_route", "wrong_criminal_route", "declaration_warnings", "declaration_repaired")


def validate_questions(questions, corpus):
    """校验每个标注的法名/条号及全部支撑词，返回错误列表；不修改输入或法规。

    每个 expect_any 元素必须有非空 support_keywords；任一候选标注失败即拒绝整题入库。
    不判断事实真伪和最终个案适用；禁止将本结构校验表述为人工核准。
    """
    by_key = defaultdict(list)
    for row in corpus:
        by_key[(row.get("title"), article_key(row.get("locator", "")))].append(row.get("text", ""))
    errors, seen = [], set()
    for q in questions:
        qid = q.get("id", "")
        if not qid or qid in seen:
            errors.append(f"{qid}: 题号缺失或重复")
        seen.add(qid)
        if (not q.get("query") or "《" in q["query"] or ARTICLE.search(q["query"])
                or not q.get("category") or not q.get("expect_scenario")
                or q.get("label_status") != DRAFT_STATUS or q.get("split") not in {"dev", "holdout"}):
            errors.append(f"{qid}: 口语问题/分类/拆分/草拟状态不符合要求")
        expected = q.get("expect_any", [])
        if not expected:
            errors.append(f"{qid}: 没有可接受标注")
        for target in expected:
            key = (target.get("title"), article_key(target.get("article_no", "")))
            words = target.get("support_keywords")
            valid_words = isinstance(words, list) and bool(words) and all(isinstance(w, str) and w.strip() for w in words)
            if key[1] is None or not valid_words or not any(all(w in text for w in words) for text in by_key.get(key, [])):
                errors.append(f"{qid}: 标注不存在或支撑词不全：{key}")
    return errors


def fixed_split(questions, seed=20261031):
    """按类别稳定分层拆分：12×10题生成80 dev/40 holdout，返回副本，无随机全局副作用。

    排序后使用独立随机源，输入顺序不改变拆分；每类至少保留3道未参与调参的题。
    """
    groups = defaultdict(list)
    for q in questions:
        groups[q["category"]].append(q)
    if len(groups) != 12 or any(len(group) != 10 for group in groups.values()):
        raise ValueError("固定拆分要求12类，每类10题")
    rng = random.Random(seed)
    categories = sorted(groups)
    rng.shuffle(categories)
    output = []
    for index, category in enumerate(categories):
        group = sorted(groups[category], key=lambda q: q["id"])
        rng.shuffle(group)
        dev_count = 7 if index < 8 else 6
        output.extend({**q, "split": "dev" if i < dev_count else "holdout"} for i, q in enumerate(group))
    return sorted(output, key=lambda q: q["id"])


def score_retrieval(question, hits, *, mode, error=None):
    """按Top-4首个可接受法名/条号计算命中和倒数排名；异常保留为零分，不移出分母。"""
    expected = {(r["title"], article_key(r["article_no"])) for r in question["expect_any"]}
    rank = next((i for i, row in enumerate(hits[:4], 1)
                 if (row["title"], article_key(row["locator"])) in expected), None)
    return dict(id=question["id"], split=question["split"], category=question["category"],
                query=question["query"], expect_any=question["expect_any"],
                hit=rank is not None and not error, reciprocal_rank=1 / rank if rank and not error else 0.0,
                rank=rank, mode=mode, error=error, top_4=hits[:4])


def aggregate_retrieval(results):
    """分别返回dev、holdout（及可选v2）整体与分类指标；空集合率为None，不伪造通过率。"""
    def stats(items):
        count = len(items)
        return dict(count=count, hits=sum(bool(r["hit"]) for r in items),
                    hit_at_4=sum(bool(r["hit"]) for r in items) / count if count else None,
                    mrr=sum(r["reciprocal_rank"] for r in items) / count if count else None,
                    errors=sum(bool(r.get("error")) for r in items),
                    mode_counts=dict(Counter(r["mode"] for r in items)))
    output = {}
    for split in ("dev", "holdout", "v2"):
        items = [r for r in results if r["split"] == split]
        output[split] = {**stats(items), "categories": {
            c: stats([r for r in items if r["category"] == c]) for c in sorted({r["category"] for r in items})}}
    return output


def compare_runs(previous, current):
    """只比较同题集、同语料、同向量模型及相同实际模式的完整运行，返回题号变化，无副作用。

    排序代码指纹允许变化，否则无法测改进；问题集合和逐题模式必须一致，避免降级伪涨跌。
    """
    empty = dict(comparable=False, newly_fixed=[], regressed=[], holdout_regressions=[])
    if not previous:
        return {**empty, "reason": "没有上一轮报告"}
    if previous.get("source_changed") or current.get("source_changed"):
        return {**empty, "reason": "不可比：运行中源码改变"}
    for key in ("dataset_sha256", "v2_sha256", "corpus_id", "embedding_id", "mode_counts"):
        if previous["scope"].get(key) != current["scope"].get(key):
            return {**empty, "reason": f"不可比：{key}不同"}
    old = {r["id"]: r for r in previous["results"]}
    new = {r["id"]: r for r in current["results"]}
    if len(old) != len(previous["results"]) or len(new) != len(current["results"]) or old.keys() != new.keys():
        return {**empty, "reason": "不可比：题号集合不同或重复"}
    if any(old[q].get("mode") != new[q].get("mode") or old[q]["split"] != new[q]["split"] for q in old):
        return {**empty, "reason": "不可比：逐题检索模式或拆分改变"}
    fixed = sorted(q for q in old if not old[q]["hit"] and new[q]["hit"])
    regressed = sorted(q for q in old if old[q]["hit"] and not new[q]["hit"])
    return dict(comparable=True, reason="同口径运行", newly_fixed=fixed, regressed=regressed,
                holdout_regressions=[q for q in regressed if new[q]["split"] == "holdout"])


def sample_answers(questions, count=20, *, seed=20261031, split="dev"):
    """按固定种子分层轮取真实推理样本，默认只用dev；数量不足拒绝，禁止悄悄补holdout。"""
    groups = defaultdict(list)
    for q in sorted(questions, key=lambda q: q["id"]):
        if split == "all" or q["split"] == split:
            groups[q["category"]].append(q)
    if not 1 <= count <= sum(map(len, groups.values())):
        raise ValueError("抽检数量超出指定拆分题量")
    rng = random.Random(seed)
    categories = sorted(groups)
    rng.shuffle(categories)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        for category in categories:
            if groups[category] and len(selected) < count:
                selected.append(groups[category].pop())
    return selected


def acceptance_gate(baseline, current):
    """对冻结基线验收：dev命中提升且MRR不降、holdout命中/MRR/逐题无退步、v2至少33题。

    不可比或未执行完整80/40拆分时拒绝给出通过；不按最新一次试验成绩掩盖相对基线的退步。
    """
    diff = compare_runs(baseline, current)
    reasons = []
    if not diff["comparable"]:
        return dict(passed=False, reasons=[diff["reason"]], baseline_comparison=diff)
    before, after = baseline["summary"], current["summary"]
    if after["dev"]["count"] != 80 or after["holdout"]["count"] != 40:
        reasons.append("仅开发子集运行，未执行完整80/40验收")
    if after["dev"]["hits"] <= before["dev"]["hits"] or after["dev"]["mrr"] < before["dev"]["mrr"]:
        reasons.append("dev命中未提升或MRR下降")
    if after["holdout"]["hits"] < before["holdout"]["hits"] or after["holdout"]["mrr"] < before["holdout"]["mrr"] or diff["holdout_regressions"]:
        reasons.append("HOLDOUT退步，必须回退候选改动")
    if after["v2"]["hits"] < 33:
        reasons.append("v2低于33/36")
    if any(r.get("error") or r.get("mode") != "hybrid_qdrant" for r in current["results"]):
        reasons.append("存在异常或检索降级，不能作为真实混合验收")
    if current.get("source_changed"):
        reasons.append("运行中源码改变")
    return dict(passed=not reasons, reasons=reasons, baseline_comparison=diff)


def aggregate_answers(checks):
    """统计可观测缺陷，异常数保留总分母；每项另列无法检查数，不把失败算作零缺陷。"""
    output = dict(requested=len(checks), completed=sum(c["completed"] for c in checks),
                  errors=sum(not c["completed"] for c in checks), checks={})
    for flag in ANSWER_FLAGS:
        known = [c[flag] for c in checks if c.get(flag) is not None]
        output["checks"][flag] = dict(numerator=sum(known), denominator=len(known),
                                     rate=sum(known)/len(known) if known else None, unknown=len(checks)-len(known))
    output["unbound_article_answers"] = sum(bool(c.get("unbound_articles")) for c in checks)
    return output


def compare_answer_runs(previous, current):
    """同题、同模型/语料的真实回答前后对比，按原始/展示分层统计；不忽略错误或改变原始回答。"""
    if not previous:
        return dict(comparable=False, reason="没有回答基线", metrics={})
    for key in ("dataset_sha256", "corpus_id", "embedding_id", "mode_counts"):
        if previous["scope"].get(key) != current["scope"].get(key):
            return dict(comparable=False, reason=f"回答来源不可比：{key}", metrics={})
    old = {r['id']: r for r in previous['results']}
    new = {r['id']: r for r in current['results']}
    if (old.keys() != new.keys() or len(old) != len(previous['results']) or len(new) != len(current['results'])
            or previous['runtime']['model_digest'] != current['runtime']['model_digest']
            or any(old[q]['query'] != new[q]['query'] or old[q]['mode'] != new[q]['mode'] for q in old)):
        return dict(comparable=False, reason="题号/问句/模式或4B模型摘要改变", metrics={})
    def measure(rows, flag):
        known = [r['checks'][flag] for r in rows if r['checks'].get(flag) is not None]
        return dict(numerator=sum(known), denominator=len(known), rate=sum(known)/len(known) if known else None,
                    unknown=len(rows)-len(known))
    return dict(comparable=True, reason="同题同模型原始/展示分层比较", metrics={
        flag: dict(before=measure(previous['results'], flag), after=measure(current['results'], flag)) for flag in ANSWER_FLAGS})


def audit_answer(question, result, materials, *, error=None):
    """检查真实raw/展示回答的可观测结构；出错返回未完成及空检查值，不当作零缺陷。

    引用只核对显式法名/条号及内部编号是否在期望集合或本轮材料中，不宣称法律推理正确。
    多步骤以处理方向段中的至少两个不同顺序标记判断；650计展示文本非空白字符。

    结构化材料回答（answer_assembly 的 selection/none/recovered）由程序装配正文，
    模型 raw 只是 SELECT/ASK；结构、声明和引用改核最终装配文本 final_answer，
    仍对 raw 单独保留“禁止机构模板/刑民路径”检查。旧格式 passthrough 仍核 raw。
    """
    if error:
        return dict(completed=False, error=error, **{flag: None for flag in ANSWER_FLAGS})
    raw, display = result.get("raw_answer", ""), result.get("answer", "")
    assembly_mode = (result.get("answer_assembly") or {}).get("mode")
    structured = assembly_mode in {"selection", "none", "recovered"}
    checked = (result.get("final_answer") or raw) if structured else raw
    allowed = {(r["title"], article_key(r["article_no"])) for r in question["expect_any"]}
    allowed |= {(r["title"], article_key(r["locator"])) for r in materials}
    by_id = {m["material_id"]: m for m in materials}
    body_ids = set(find_citations(checked))
    declared_ids = set(parse_used_ids(checked) or [])
    unknown_ids = (body_ids | declared_ids) - by_id.keys()
    undeclared_ids = body_ids - declared_ids
    bad_named = []
    for match in re.finditer(r"《([^》]+)》\s*(第[零〇一二三四五六七八九十百千万两\d]+条(?:之[一二三四五六七八九十\d]+)?)", checked):
        name = match[1]
        if not any(article_key(match[2]) == key and name in {title, title.removeprefix("中华人民共和国")} for title, key in allowed):
            bad_named.append(match[0])
    # 无法绑定法名的裸条号不悄悄算通过，单列待核验；紧邻材料编号时可逐项绑定。
    unbound = []
    for match in ARTICLE.finditer(checked):
        before = checked[:match.start()].split("\n")[-1]
        if re.search(r"《[^》]+》\s*$", before):
            continue
        mids = re.findall(r"M\s*(\d+)", checked[max(0, match.start()-12):match.end()+12])
        if not any(f"M{mid}" in by_id and article_key(by_id[f"M{mid}"]["locator"]) == article_key(match[0]) for mid in mids):
            unbound.append(match[0])
    directions = re.search(r"(?m)^##\s*处理方向\s*\n([\s\S]*?)(?=^##\s|\Z)", checked)
    steps = re.findall(r"(?:^|[\n；;。\s])(?:[（(]?([123一二三])[)）、.．]|第([一二三123])步)", directions[1] if directions else "")
    step_ids = {(a or b).translate(str.maketrans("一二三", "123")) for a, b in steps}
    length = len(re.sub(r"\s", "", display))
    # 复用独立显式引用审计，补上无书名号简称、条号与材料编号错绑；允许集合仍只取题目标注/真实材料。
    reference_rows = [{**m, "version_id": m.get("version_id", "")} for m in materials]
    reference_rows += [dict(material_id=f"EXPECT-{n}", title=t["title"], locator=t["article_no"], version_id="")
                       for n, t in enumerate(question["expect_any"])
                       if (t["title"], article_key(t["article_no"])) not in {(m["title"], article_key(m["locator"])) for m in materials}]
    citation_audit = audit_citations(checked, reference_rows, reference_rows)
    # 书名号也可括文书名（如催告函），不能仅因后文出现“要求”就计作伪造法律。
    # 明确法条错绑仍由上方条号检查保留；无条号但名称呈法规结构的未知法律仍记缺陷。
    citation_issues = [issue for issue in citation_audit["issues"]
                       if issue["type"] != "UNKNOWN_LAW_TITLE"
                       or ARTICLE.fullmatch(issue["value"])
                       or re.search(r"(?:法律|法|条例|规定|办法|解释|细则|规则|决定|通知|意见|标准|规范)(?:[（(][^）)]*[）)])?$", issue["value"])]
    def wrong_route(text):
        # 独立评测规则，不调用生产拦截函数；只度量显式反诉误用，不声称完整管辖语义审计。
        criminal = question['expect_scenario'] == '街头刑事' or re.search(r'强奸|猥亵|诬告|性侵|刑事', question['query'])
        if not criminal:
            return False
        civil_case = bool(re.search(r'民事(?:诉讼|起诉|案件)|(?:借款|欠款|货款|合同).{0,12}起诉', question['query']))
        for clause in re.split(r'[。！？；;\n，,]', text):
            if civil_case and re.search(r'借款|欠款|货款|合同|民事案件', clause) and not re.search(r'强奸|猥亵|诬告|诈骗|刑事|犯罪', clause):
                continue
            for match in re.finditer('反诉|民事反诉', clause):
                prefix = clause[:match.start()].split('但')[-1]
                if not re.search(r'(?:不能|不得|不应当?|不要|不宜|严禁|禁止|避免|不属于|不建议|无需)'
                                 r'(?:再|直接)?(?:以|用|通过)?(?:提起|提出|进行|采取|申请|启动)?[ \t“"「]*$', prefix):
                    return True
        return False
    unrelated = question['expect_scenario'] not in {'单位', '校园'}
    return dict(completed=True, error=None, blocked=bool(result.get("blocked")),
                missing_used_ids=parse_used_ids(checked) is None,
                display_internal_ids=bool(re.search(r"\[\s*M\s*\d+\s*\]|USED_IDS\s*[:：]", display)),
                missing_headings=not all(re.search(rf"(?m)^##\s*{h}\s*$", checked) for h in HEADINGS),
                missing_steps=len(step_ids) < 2,
                wrong_institutions=question["expect_scenario"] not in {"单位", "校园"} and any(x in display for x in INSTITUTIONS),
                raw_wrong_institutions=unrelated and any(x in raw for x in INSTITUTIONS),
                raw_wrong_criminal_route=wrong_route(raw), wrong_criminal_route=wrong_route(display),
                declaration_warnings=any(w.get('type') in {'MISSING_USED_IDS', 'ID_CITED_BUT_NOT_DECLARED'} for w in result.get('warnings', [])),
                declaration_repaired=bool(result.get('declaration_repair', {}).get('repaired')),
                bad_citations=bool(unknown_ids or undeclared_ids or bad_named or citation_issues),
                citation_issues=citation_issues, unbound_articles=unbound,
                unknown_ids=sorted(unknown_ids), undeclared_ids=sorted(undeclared_ids), bad_named=bad_named,
                over_650=length > 650, display_characters=length)
