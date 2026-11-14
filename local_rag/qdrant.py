"""Qdrant 本地向量数据库适配模块
职责：
1. 负责连接本地 Qdrant 实例、创建或管理集合（Collection）。
2. 将切分后的条文数据与对应的向量以批次形式写入 Qdrant。
3. 执行余弦相似度检索，返回最匹配的资料及相似度得分。
注意：本模块只负责向量存储与检索交互，不负责向量生成（向量由 OllamaClient 生成）。

安全与隔离原则：
- 与用户本地已有数据库共用实例时，只操作以 legal_rag_ 开头的项目集合，绝不清空或影响其他集合。
- 按语料指纹（corpus_id）严格隔离批次：新版未完整写入并校验通过前，绝不混用旧版条文或其他项目的向量。

阅读指南：docs/guides/PYCHARM_PROJECT_READING.md
"""
import hashlib
import json
import os
import re
import uuid
from urllib.parse import urlsplit

import httpx

# 文档向量化格式（默认 plain 与改造前逐字节一致）；集合名与批次指纹都带上格式标签。
PLAIN_DOC_FORMAT = 'plain'


def collection_name(model_id, size, doc_format=PLAIN_DOC_FORMAT):
    """返回集合名；plain 沿用旧命名，其他格式在模型哈希后追加格式标签，避免新旧向量混住。"""
    if not re.fullmatch(r'[a-z0-9_]+', doc_format):
        raise ValueError('文档格式标签只能包含小写字母、数字和下划线')
    identity = model_id if doc_format == PLAIN_DOC_FORMAT else model_id + '\n' + doc_format
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    tag = '' if doc_format == PLAIN_DOC_FORMAT else '_' + doc_format
    return 'legal_rag_' + digest + tag + '_' + str(size)


def corpus_fingerprint(model_id, rows, doc_format=PLAIN_DOC_FORMAT):
    """返回语料批次指纹；plain 保持旧口径，其他格式把格式名纳入哈希，新旧批次严格隔离。"""
    payload = [model_id, rows] if doc_format == PLAIN_DOC_FORMAT else [model_id, doc_format, rows]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class QdrantStore:
    """Qdrant 向量数据库本地客户端适配器
    
    仅通过本机 REST 接口访问 Qdrant；当数据库不可用或网络异常时，由上层调度模块（Retriever）明确降级为关键词检索。
    """

    def __init__(self, url='http://127.0.0.1:16333', *, transport=None):
        """初始化 Qdrant 客户端连接
        
        参数说明：
            url: Qdrant 服务的 HTTP 访问地址，默认为本机环回端口 http://127.0.0.1:16333。
            transport: 可选的自定义 HTTP 传输层，主要用于单元测试注入 Mock 请求。
            
        安全校验：
            严格限制仅允许本机 HTTP 环回地址（localhost/127.0.0.1/::1），杜绝外部网络外发风险。
        """
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
                or parsed.username or parsed.password or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
            raise ValueError('向量数据库仅允许本机HTTP环回地址')
        
        headers = {}
        if os.getenv('QDRANT_API_KEY'):
            headers['api-key'] = os.environ['QDRANT_API_KEY']
            
        self.http = httpx.AsyncClient(
            base_url=url.rstrip('/'),
            headers=headers,
            trust_env=False,
            follow_redirects=False,
            timeout=120,
            transport=transport
        )
        self.collection = ''   # 当前生效的集合名称
        self.corpus_id = ''    # 当前语料批次的唯一哈希指纹
        self.ready = False     # 当前批次是否完成 sync 入库校验；单次查询异常不改变此标志。
        self.rows = []         # 内存中缓存的条文数据列表
        self.count = 0         # 成功写入数据库的点（Point）总数

    async def request(self, method, path, body=None):
        """发送底层 REST 请求给 Qdrant
        
        参数说明：
            method: HTTP 方法（GET, POST, PUT, DELETE）。
            path: 请求路径（例如 /collections/xxx）。
            body: 请求体字典（JSON）。
            
        返回值：
            Qdrant 返回 JSON 中的 'result' 字段数据。
        异常说明：
            响应非 2xx 时抛出 HTTPStatusError，不暴露完整认证或私密数据。
        """
        response = await self.http.request(method, path, json=body)
        response.raise_for_status()
        return response.json()['result']

    def scope(self):
        """构建 Qdrant 查询过滤条件（Filter）
        
        用途：
            在检索时强制限定 corpus_id 必须等于当前加载语料批次的唯一指纹。
            避免因同集合多次增量入库或历史残留，将旧版条文或其他无关项目的数据错误召回。
        """
        return {'must': [{'key': 'corpus_id', 'match': {'value': self.corpus_id}}]}

    async def sync(self, rows, vectors, model_id, doc_format=PLAIN_DOC_FORMAT):
        """将全量语料和计算好的向量幂等写入 Qdrant，并严格校验写入点数
        
        参数说明：
            rows: 结构化条文字典列表（包含 id, title, locator, text 等字段）。
            vectors: 已经归一化好的稠密向量列表（每个元素为浮点数数组）。
            model_id: 生成该批向量的模型标识（例如 qwen3-embedding:0.6b），用于隔离不同模型的集合。
            doc_format: 文档向量化格式版本；默认 plain 保持旧集合名和旧 corpus_id 不变。
            
        核心机制：
            1. 集合命名规则：legal_rag_{模型哈希前12位}_{格式标签}_{向量维度}；plain 省略格式标签。
            2. 批次哈希指纹：根据 model_id、文档格式和全量 rows 内容生成唯一的 corpus_id。
            3. 点的唯一标识（UUID5）：通过 corpus_id + row['id'] 生成确定性 UUID，即便多次调用也是覆盖更新，绝不产生重复点。
            4. 分块写入：按每批 64 条批量 PUT 入库。
            5. 精确核验：写入完成后调用 points/count，若数量与 rows 不相等则拒绝标记 ready。
        """
        self.ready = False
        if not rows or len(rows) != len(vectors) or len({len(v) for v in vectors}) != 1:
            raise RuntimeError('资料与向量数量或维数不一致，未写入数据库')
        
        size = len(vectors[0])
        # 集合名包含模型指纹、文档格式和向量维度，避免模型/格式更换时混用旧向量
        self.collection = collection_name(model_id, size, doc_format)
        self.corpus_id = corpus_fingerprint(model_id, rows, doc_format)
        path = '/collections/' + self.collection
        
        # 检查集合是否存在，不存在则新建（使用余弦相似度 Cosine）
        response = await self.http.get(path)
        if response.status_code == 404:
            await self.request('PUT', path, {'vectors': {'size': size, 'distance': 'Cosine'}})
        else:
            response.raise_for_status()
            config = response.json()['result']['config']['params']['vectors']
            if config.get('size') != size or config.get('distance') != 'Cosine':
                raise RuntimeError('已有项目集合维数/距离不兼容；未重建或清空，请核对配置')
        
        # 检查当前语料批次是否已经写入过
        existing = await self.request('POST', path + '/points/count', {'exact': True, 'filter': self.scope()})
        if existing['count'] != len(rows):
            # 分批（每批 64 条）上传向量和条文元数据
            for offset in range(0, len(rows), 64):
                points = []
                for row, vector in zip(rows[offset:offset + 64], vectors[offset:offset + 64]):
                    # UUIDv5 由批次指纹与条文唯一ID共同派生，实现确定性幂等写入
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, self.corpus_id + '/' + row['id']))
                    payload = dict(
                        row,
                        chunk_id=row['id'],
                        corpus_id=self.corpus_id,
                        embedding_model=model_id,
                        project='legal-rag'
                    )
                    if doc_format != PLAIN_DOC_FORMAT:
                        # 只给新格式点加标记，便于排查；plain 旧点保持原 payload。
                        payload['doc_format'] = doc_format
                    points.append({'id': point_id, 'vector': vector, 'payload': payload})
                await self.request('PUT', path + '/points?wait=true', {'points': points})
        
        # 最终精确校验写入数量
        counted = await self.request('POST', path + '/points/count', {'exact': True, 'filter': self.scope()})
        if counted['count'] != len(rows):
            raise RuntimeError('数据库批次数量核对失败，未启用向量查询')
            
        self.rows = rows
        self.count = counted['count']
        self.ready = True

    async def search(self, vector, limit=12):
        """在当前语料批次中搜索与查询向量最相似的候选条文
        
        参数说明：
            vector: 用户问题的归一化查询向量。
            limit: 最大召回条目数（默认 12 条）。
            
        返回值：
            形如 [(资料内存下标, 相似度得分), ...] 的列表，按相似度降序排序。
            
        安全机制：
            召回的 payload 必须逐字比对本地内存中的原始文本，若出现不一致则拒绝作为大模型材料，杜绝被篡改。
        """
        if not self.ready:
            raise RuntimeError('当前语料尚未完整写入Qdrant')
            
        result = await self.request(
            'POST',
            '/collections/' + self.collection + '/points/query',
            {
                'query': vector,
                'filter': self.scope(),
                'limit': limit,
                'score_threshold': 0.38,   # 过滤余弦相似度过低的条目
                'with_payload': True
            }
        )
        
        by_id = {row['id']: i for i, row in enumerate(self.rows)}
        hits = []
        for point in result['points']:
            payload = point['payload']
            idx = by_id.get(payload.get('chunk_id'))
            
            # 双重核验：确保召回的确实是本地对应批次的同名原始条文
            if (idx is None or payload.get('corpus_id') != self.corpus_id
                    or any(payload.get(k) != v for k, v in self.rows[idx].items())):
                raise RuntimeError('数据库召回内容与本地核验资料不一致，已拒绝作为模型材料')
            hits.append((idx, point['score']))
            
        return hits

    def status(self):
        """返回 Qdrant 当前运行与连接状态，供 Web 前端或监控器展示真实数据库情况"""
        return {
            'backend': 'qdrant',
            'ready': self.ready,
            'collection': self.collection,
            'points': self.count,
            'corpus_id': self.corpus_id
        }

    async def close(self):
        """优雅关闭 HTTP 客户端连接（不会停止容器或删除数据）"""
        await self.http.aclose()

