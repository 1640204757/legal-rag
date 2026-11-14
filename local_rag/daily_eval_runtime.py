"""日常评测的真实运行与留证：只读Qdrant、调用本机模型、保存独立评测结果。

主入口 open_runtime 读取Web状态并逐字核验语料指纹；不调用build/sync、不修改索引。
load_dataset负责入库标注校验；save_report只写指定评测目录，拒绝覆盖已有运行。
"""
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import httpx

from .daily_evaluation import compare_runs, fixed_split, validate_questions
from .ollama import OllamaClient
from .qdrant import QdrantStore, collection_name, corpus_fingerprint
from .retrieval import DOC_FORMAT_PLAIN, DOC_FORMATS, ROOT, Retriever, load_corpus


def sha256_file(path):
    """返回文件原始字节哈希；只读，用于冻结题集与代码来源留证。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path, rows):
    """读入v3题集并核验每个标注、固定拆分、原文哈希及旧题集保护哈希；失败拒绝运行。"""
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    questions = spec["questions"]
    errors = validate_questions(questions, rows)
    frozen = json.loads(path.with_suffix(".frozen.json").read_text(encoding="utf-8"))
    if sha256_file(path) != frozen["sha256"]:
        errors.append("v3冻结题集哈希不匹配，不能按成绩换题/改标签")
    if questions != fixed_split(questions, spec["seed"]):
        errors.append("题目拆分与固定种子不符")
    by_key = {(r["title"], r["locator"].split()[0]): sha256_file_bytes(r["text"]) for r in rows}
    for q in questions:
        for target in q["expect_any"]:
            if target.get("text_sha256") != by_key.get((target["title"], target["article_no"])):
                errors.append(f"{q['id']}: 标注原文哈希变更，须重新审核而非静默改绑")
    for name, digest in spec["protected_eval_sha256"].items():
        if Path(name).name != name or sha256_file(path.parent / name) != digest:
            errors.append(f"旧评测文件变更：{name}")
    if errors:
        raise ValueError("评测集校验失败：" + "; ".join(errors))
    return spec, sha256_file(path)


def sha256_file_bytes(text):
    """计算UTF-8文本指纹，不访问文件；与条文入库时保存的哈希口径相同。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReadOnlyQdrant(QdrantStore):
    """评测只读适配器：拒绝写库请求，记录每题实际通过阈值及原文核验的向量候选数。"""
    qualified_count = 0

    async def request(self, method, path, body=None):
        """仅放行集合状态、精确计数及检索请求；写操作在网络前拒绝。"""
        if not (method == "GET" or method == "POST" and path.endswith(("/points/count", "/points/query"))):
            raise RuntimeError("评测禁止写入Qdrant")
        return await super().request(method, path, body)

    async def search(self, vector, limit=20):
        """透传生产原文核验，记录实际合格候选数；不以服务ready状态代替成功检索。"""
        self.qualified_count = 0
        hits = await super().search(vector, limit)
        self.qualified_count = sum(score >= .38 for _, score in hits)
        return hits


async def actual_search(retriever, question, *, top_k=4):
    """按调用方条数执行生产检索（默认前4）；无合格向量时标记降级，不修改生产打分。"""
    if retriever.store:
        retriever.store.qualified_count = 0
    hits, mode = await retriever.search(question, top_k=top_k)
    if mode == "hybrid_qdrant" and not retriever.store.qualified_count:
        mode = "lexical_no_qualified_dense"
    return hits, mode


def loopback_url(value):
    """校验评测本机服务地址，拒绝远端、凭证、路径及重定向入口；返回规范基础地址。"""
    url = urlsplit(value)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username is not None or url.password is not None or url.path not in {"", "/"}
            or url.query or url.fragment):
        raise ValueError("评测仅允许本机HTTP环回服务")
    return value.rstrip("/")


@asynccontextmanager
async def open_runtime(*, api_url="http://127.0.0.1:18765", qdrant_url="http://127.0.0.1:16333",
                       ollama_url="http://127.0.0.1:11434", lexical=False,
                       doc_format=DOC_FORMAT_PLAIN, collection=None):
    """连接当前Web声明的集合/语料批次，返回生产检索器、模型及脱敏来源信息。

    仅查询现有集合，不生成资料向量、不写磁盘或数据库；关闭时释放本机HTTP连接。
    服务/指纹/计数不匹配会明确降级词面并记录原因，绝不偷用别批向量。

    doc_format 非 plain 或显式传入 collection 时，用于只读测量实验集合：
    模型标签仍取自 Web，但集合名与 corpus_id 按本地语料和文档格式重算并严格核验。
    """
    api_url, qdrant_url, ollama_url = map(loopback_url, (api_url, qdrant_url, ollama_url))
    if doc_format not in DOC_FORMATS:
        raise ValueError(f"未知文档向量化格式：{doc_format}")
    rows, coverage = load_corpus()
    model = OllamaClient(ollama_url)
    store = ReadOnlyQdrant(qdrant_url)
    retriever = Retriever(rows, model, store=store, doc_format=doc_format)
    info = dict(api_url=api_url, coverage=coverage, collection="", corpus_id="", embedding_id="",
                model="", model_digest="", degradation_reason="", write_database=False,
                doc_format=doc_format)
    try:
        if lexical:
            info["degradation_reason"] = "明确指定纯词面评测，不计作混合成绩"
        else:
            try:
                async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=10) as client:
                    response = await client.get(api_url + "/api/status")
                    response.raise_for_status()
                    status = response.json()
                database = status["database"]
                if not database.get("ready") or status.get("index", {}).get("state") != "ready":
                    raise ValueError("Web索引或Qdrant未就绪")
                declared_collection, declared_corpus = database["collection"], database["corpus_id"]
                if (not re.fullmatch(r"legal_rag_[A-Za-z0-9_]+", declared_collection)
                        or not re.fullmatch(r"[0-9a-f]{64}", declared_corpus)):
                    raise ValueError("Web集合/指纹格式不符合项目范围")
                model.model, model.embedding = status["model"], status["embedding"]
                # 用构造器的同一边界校验模型标签，避免错误Web配置指向云端模型。
                if any(x in tag.lower() for tag in (model.model, model.embedding) for x in ("cloud", "http", "/", "\\")):
                    raise ValueError("非本机模型标签")
                tags_response = await model.http.get("/api/tags", timeout=10)
                tags_response.raise_for_status()
                tags = tags_response.json()["models"]
                embedding_digest = next(m["digest"] for m in tags if m["name"] == model.embedding)
                model.model_id = model.embedding + ":" + embedding_digest
                # plain 默认仍以 Web 声明的生产集合为准；实验格式必须显式指定新集合。
                selected = collection or declared_collection
                if doc_format == DOC_FORMAT_PLAIN and not collection:
                    selected = declared_collection
                elif doc_format == DOC_FORMAT_PLAIN and collection != declared_collection:
                    raise ValueError("plain 格式只能连接 Web 声明的生产集合")
                # 指纹一律按本地语料和当前文档格式重算；plain 与旧公式逐字节相同。
                actual_id = corpus_fingerprint(model.model_id, rows, doc_format)
                if selected == declared_collection and declared_corpus != actual_id:
                    raise ValueError("当前磁盘语料/模型指纹与Web不一致")
                if selected == declared_collection and database.get("points") != len(rows):
                    raise ValueError("当前磁盘语料段数与Web不一致")
                store.collection, store.corpus_id, store.rows = selected, actual_id, rows
                collection_info = await store.request("GET", "/collections/" + selected)
                dimension = collection_info["config"]["params"]["vectors"]["size"]
                if selected != collection_name(model.model_id, dimension, doc_format):
                    raise ValueError("集合名与模型/文档格式/向量维数不一致")
                counted = await store.request("POST", "/collections/" + selected + "/points/count",
                                              {"exact": True, "filter": store.scope()})
                if counted["count"] != len(rows) or not isinstance(dimension, int) or not 1 <= dimension <= 8192:
                    raise ValueError("Qdrant当前批次精确计数或向量维数不符")
                store.ready, store.count = True, counted["count"]
                # 生产search在Qdrant路径只读取此维数，不读取资料向量值；避免载入数百MB缓存或触发build。
                retriever.vectors = [[1.0] + [0.0] * (dimension - 1)]
                retriever.state = "ready"
                info.update(collection=selected, corpus_id=actual_id, embedding_id=model.model_id,
                            model=model.model, model_digest=next((m["digest"] for m in tags if m["name"] == model.model), "missing"))
            except (httpx.HTTPError, ValueError, KeyError, TypeError, StopIteration) as exc:
                info["degradation_reason"] = f"真实混合不可用，降级纯词面：{type(exc).__name__}: {str(exc)[:180]}"
                retriever.vectors = None
        if not info["corpus_id"]:
            info["corpus_id"] = "lexical:" + sha256_file_bytes(json.dumps(rows, ensure_ascii=False, sort_keys=True))
        yield retriever, model, info
    finally:
        await model.close()
        await store.close()


def make_scope(info, dataset_hash, v2_hash, results):
    """构造可比口径和实际检索模式计数；不包含Web令牌或用户历史。"""
    return dict(dataset_sha256=dataset_hash, v2_sha256=v2_hash, corpus_id=info["corpus_id"],
                embedding_id=info["embedding_id"], collection=info["collection"],
                retrieval_code_sha256=info.get("source_hashes", {}).get("local_rag/retrieval.py", sha256_file(ROOT / "local_rag/retrieval.py")),
                mode_counts=dict(Counter(r["mode"] for r in results)))


def execution_sources(*extra_paths):
    """冻结当前执行涉及的生产/评分源码哈希，供运行前后核对；只读，不复制或修改源文件。"""
    paths = ("local_rag/answer_permission.py", "local_rag/retrieval.py", "local_rag/service.py", "local_rag/daily_evaluation.py", "local_rag/daily_eval_runtime.py",
             "prototype/rag/citation_check.py", "prototype/rag/postprocess.py", "prototype/rag/grounding.py", *extra_paths)
    return {p: sha256_file(ROOT / p) for p in paths}


def prompt_materials(prompt):
    """从真实RagInput消息恢复已送模完整材料，供生成出错时留证；不猜Top-4或截断补齐。

    只读取【检索材料】与【覆盖缺口】之间的生产渲染格式；边界缺失则返回空。
    """
    section = re.search(r"【检索材料】\n([\s\S]*?)(?:\n【覆盖缺口】|\Z)", prompt)
    if not section:
        return []
    pattern = re.compile(r"(?m)^\[(M\d+)\] ([^｜\n]+)｜([^｜\n]+)｜([^｜\n]+)｜版本 ([^\n]+)\n"
                         r"来源：([^\n]*)\n核验范围：([^\n]*)(?:\n注意：[^\n]*)?\n原文：([\s\S]*?)(?=\n\n\[M\d+\] |\Z)")
    # 结构化选择模式会把〔M1#1〕内联在句首；留证时去掉标记，恢复成原始材料文本。
    return [dict(material_id=m[1], kind=m[2], title=m[3], locator=m[4], version_id=m[5], source_url=m[6],
                 verified_scope=m[7], text=re.sub(r"〔M\d+#\d+〕", "", m[8]).rstrip("\n"))
            for m in pattern.finditer(section[1])]


def save_report(report, output_root):
    """创建唯一运行目录并保存JSON/可读报告；自动与上次同类运行比较，不覆盖历史结果。

    不可比时记录原因，不跳过最近的降级运行去挑选有利历史成绩。返回新目录路径。
    Windows同一时钟刻度撞名时递增序号，沿用测量尺子的原子建目录方式，不覆盖旧报告。
    """
    output_root = Path(output_root)
    kind = report["kind"]
    previous_paths = sorted(output_root.glob(f"*/{kind}.json"))
    previous = json.loads(previous_paths[-1].read_text(encoding="utf-8")) if previous_paths else None
    report = {**report, "comparison": compare_runs(previous, report) if kind == "retrieval" else None,
              "previous_report": str(previous_paths[-1]) if previous_paths else None}
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    directory, n = output_root / stamp, 1
    while True:
        try:
            directory.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            n += 1
            directory = output_root / f"{stamp}-{n}"
    report["run_directory"] = str(directory)
    (directory / f"{kind}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# 日常评测：{kind}", "", f"运行：{directory.name}",
             "标注：AI草拟待人工抽检；程序支撑词校验不等于法律适用审核。", "",
             f"耗时：{report.get('elapsed_seconds', 0):.2f}秒", f"来源：{json.dumps(report.get('scope', {}), ensure_ascii=False)}",
             f"降级说明：{report.get('degradation_reason') or '无；实际模式另逐题记录'}", ""]
    for split, stats in report.get("summary", {}).items():
        lines += [f"## {split}", json.dumps(stats, ensure_ascii=False, indent=2), ""]
    if report.get("acceptance") is not None:
        lines += ["## 相对冻结基线验收", json.dumps(report["acceptance"], ensure_ascii=False, indent=2), ""]
    if report.get('answer_comparison') is not None:
        lines += ['## 回答原始/展示分层对比', json.dumps(report['answer_comparison'], ensure_ascii=False, indent=2), '']
    diff = report["comparison"]
    if diff:
        lines += ["## 与上次运行比较", diff["reason"], f"新修好：{diff['newly_fixed']}", f"新退步：{diff['regressed']}",
                  f"**HOLDOUT退步警报：{diff['holdout_regressions']}**", ""]
    for row in report.get("results", []):
        if kind == "retrieval" and not row["hit"]:
            lines += [f"### 未命中 {row['id']} ({row['split']}/{row['category']})", row["query"],
                      "期望：" + json.dumps(row["expect_any"], ensure_ascii=False),
                      "实际Top-4：" + json.dumps([{k: r.get(k) for k in ("title", "locator", "retrieval_score")} for r in row["top_4"]], ensure_ascii=False),
                      f"模式/异常：{row['mode']} / {row.get('error')}", ""]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory
