"""一次性全量重建“文档格式”向量索引：python scripts/build_doc_format_index.py

用途：
    把同一批 46,701 段资料按指定文档格式重新向量化并写入独立 Qdrant 集合，
    用于比较“正文前加《法名》条号”是否改善检索。只读本地资料，不调用回答模型。

安全边界：
    - 默认只允许非 plain 格式；plain 是生产格式，禁止用本脚本重建，避免误覆盖现有集合。
    - 新格式使用独立缓存文件、独立集合名和带格式的 corpus_id，旧集合与 embeddings.json 不删不改。
    - 脚本本身只写新集合和新缓存文件，不切换 Web 默认，也不重启 18765 服务。

示例：
    python scripts/build_doc_format_index.py --doc-format law_header_v1
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

from local_rag.ollama import OllamaClient
from local_rag.qdrant import QdrantStore
from local_rag.retrieval import DOC_FORMAT_PLAIN, DOC_FORMATS, Retriever, load_corpus


async def run(args):
    """加载资料、全量向量化并写入新集合；返回进程退出码。"""
    if args.doc_format == DOC_FORMAT_PLAIN:
        raise SystemExit("禁止用本脚本重建生产 plain 格式；生产入库请使用 python -m local_rag.index")

    rows, coverage = load_corpus()
    model = OllamaClient(args.ollama_url, embedding=args.embedding)
    store = QdrantStore(args.qdrant_url)
    retriever = Retriever(rows, model, args.cache_path, store=store, doc_format=args.doc_format)
    started = time.perf_counter()
    try:
        status = await model.status()
        if not status.get("embedding_ready"):
            raise SystemExit(f"本机未就绪的向量模型：{args.embedding}")
        print(json.dumps({
            "chunks": len(rows),
            "coverage": coverage,
            "doc_format": args.doc_format,
            "embedding": args.embedding,
            "model_id": model.model_id,
            "cache_path": str(retriever.cache_path),
        }, ensure_ascii=False, indent=2), flush=True)

        await retriever.build()
        elapsed = round(time.perf_counter() - started, 2)
        result = {
            "doc_format": args.doc_format,
            "state": retriever.state,
            "error": retriever.error,
            "completed": retriever.completed,
            "elapsed_seconds": elapsed,
            "cache_path": str(retriever.cache_path),
            "cache_bytes": retriever.cache_path.stat().st_size if retriever.cache_path.is_file() else 0,
            "database": store.status(),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if retriever.state == "ready" else 1
    finally:
        await model.close()
        await store.close()


def main(argv=None):
    """解析参数并执行；只接受已登记的文档格式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-format", choices=DOC_FORMATS, default="law_header_v1",
                        help="要重建的文档向量化格式；plain 被拒绝")
    parser.add_argument("--embedding", default="qwen3-embedding:0.6b")
    parser.add_argument("--cache-path", type=Path, help="可选缓存文件；默认 data/local_index/embeddings.<格式>.json")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:16333")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
