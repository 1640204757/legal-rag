"""本地资料统一入库命令行入口：python -m local_rag.index

职责：
1. 作为独立于 Web 界面的本地知识库入库/索引更新执行器。
2. 调度执行：加载本地已保存资料 → 核验原文快照 → 复用/生成条文向量 → 写入本地 Qdrant 数据库。
3. 严格执行写入后点数核对，若指定 --search 参数，还将执行一次端到端的测试检索。

前置条件：
- 资料需先通过采集脚本（ingest.py 或 catalog_ingest.py）下载、切分并保存到 data/ 目录；
- Ollama 服务必须处于运行状态（用于向量生成）；
- Docker 中的 Qdrant 服务必须正常监听（用于向量存储）。

阅读指南：docs/guides/PYCHARM_PROJECT_READING.md
"""
import argparse
import asyncio
import json
import os
from .ollama import OllamaClient
from .qdrant import QdrantStore
from .retrieval import Retriever, load_corpus


async def run(question=None):
    """异步执行全量资料加载、向量化计算与 Qdrant 数据库同步
    
    参数说明：
        question: 可选的测试检索问题字符串。如果提供，入库完成后会立即执行一次真实检索测试。
        
    执行流程：
        1. 连接本机 Ollama，检查 embedding 模型是否就绪。
        2. 调用 load_corpus()，从磁盘加载全部通过校验的法规条文与案例摘要。
        3. 构建 Retriever 检索协调器，调用 build() 生成向量缓存并同步写入 Qdrant。
        4. 确认 retriever 状态为 'ready'，否则抛出明确错误。
        5. 如传入测试 question，执行两路混合检索并打印召回结果。
        6. 输出当前资料覆盖统计（规范数、案例数、总切片数）和数据库点数报告。
    """
    model = OllamaClient()
    store = QdrantStore(os.getenv('QDRANT_LOCAL_URL', 'http://127.0.0.1:16333'))
    try:
        # 1. 检查本地模型连接
        await model.status()
        
        # 2. 从磁盘加载所有合格资料并校验哈希
        rows, coverage = load_corpus()
        
        # 3. 创建检索器并执行向量构建与数据库写入
        retriever = Retriever(rows, model, store=store)
        await retriever.build()
        if retriever.state != 'ready':
            raise RuntimeError(retriever.error)
            
        result = {'coverage': coverage, 'database': store.status()}
        
        # 4. 可选的自检检索测试
        if question:
            hits, mode = await retriever.search(question)
            result['search'] = {
                'mode': mode,
                'hits': [{'title': r['title'], 'locator': r['locator']} for r in hits]
            }
            
        # 5. 格式化输出最终状态结果
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        # 保证连接无论成功失败都能安全关闭
        await model.close()
        await store.close()


def main():
    """解析命令行参数并启动入库流程
    
    命令行示例：
        python -m local_rag.index                                   # 仅执行增量入库与数据核对
        python -m local_rag.index --search "上班路上被车撞算不算工伤" # 入库后顺带做一次真实检索自检
    """
    parser = argparse.ArgumentParser(description='校验资料并增量入库到本机Qdrant')
    parser.add_argument('--search', help='入库后执行一次检索自检，打印召回法条，不调用问答大模型')
    args = parser.parse_args()
    asyncio.run(run(args.search))


if __name__ == '__main__':
    main()

