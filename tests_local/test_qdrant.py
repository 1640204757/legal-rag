"""Qdrant 数据库本地适配协议与隔离机制自动化测试

用途说明：
验证 local_rag/qdrant.py 中的底层数据库交互行为，使用 httpx.MockTransport 进行协议级模拟，不产生真实磁盘或网络IO。

核心测试覆盖点：
1. 幂等性与增量写入：相同法条重复同步不会产生多余点（Point），修改条文后自动生成新批次指纹。
2. 批次隔离：检索时只能查到当前批次的数据，历史旧批次的废弃条文绝不被召回。
3. 集合保护：已有集合如果向量维度不匹配，必须报错并拒绝覆盖，防止破坏用户其他项目的数据。
"""
import asyncio
import hashlib
import json
import httpx
import pytest
from local_rag.qdrant import QdrantStore, collection_name, corpus_fingerprint


class Database:
    """Mock Qdrant 内存数据库：在内存中模拟 Qdrant 的 REST API，响应 points/count, points/query 等请求"""
    def __init__(self):
        self.points = {}       # 存放点记录字典：{point_id: point_data}
        self.config = None     # 存放集合的配置信息（维度、距离度量）
        self.last_query = None # 记录最近一次查询请求的体数据

    def handle(self, req):
        """处理模拟 HTTP 请求"""
        import json
        data = json.loads(req.content) if req.content else {}
        path = req.url.path
        
        # 1. 模拟统计点数接口 /points/count
        if path.endswith('/points/count'):
            corpus = data['filter']['must'][0]['match']['value']
            # 只统计属于指定 corpus_id 的点
            result = {'count': sum(p['payload']['corpus_id'] == corpus for p in self.points.values())}
            
        # 2. 模拟向量相似度查询接口 /points/query
        elif path.endswith('/points/query'):
            self.last_query = data
            corpus = data['filter']['must'][0]['match']['value']
            # 严格过滤只返回当前 corpus_id 的点
            result = {
                'points': [
                    dict(p, score=0.9)
                    for p in self.points.values()
                    if p['payload']['corpus_id'] == corpus
                ][:data['limit']]
            }
            
        # 3. 模拟插入点数据接口 /points
        elif path.endswith('/points'):
            self.points.update({p['id']: p for p in data['points']})
            result = {'status': 'completed'}
            
        # 4. 模拟集合不存在时的 404
        elif req.method == 'GET' and self.config is None:
            return httpx.Response(404, json={'status': 'error'})
            
        # 5. 模拟创建集合 PUT /collections/xxx
        elif req.method == 'PUT':
            self.config = data
            result = True
            
        # 6. 获取集合当前信息
        else:
            result = {'config': {'params': self.config}, 'points_count': len(self.points)}
            
        return httpx.Response(200, json={'status': 'ok', 'result': result})


def row(text='工伤条文'):
    """生成一条标准的法条测试记录"""
    return {
        'id': 'law/18',
        'text': text,
        'title': '测试规范',
        'source_url': 'https://www.court.gov.cn/'
    }


def test_sync_is_idempotent_and_old_corpus_not_searched():
    """测试 Qdrant 同步的幂等性与批次隔离防御机制"""
    async def run():
        db = Database()
        # 注入 Mock 传输层
        store = QdrantStore(transport=httpx.MockTransport(db.handle))
        
        # 第一次同步写入
        a = [row()]
        await store.sync(a, [[1.0, 0.0]], 'embedding-v1')
        old_corpus = store.corpus_id
        
        # 重复同步完全相同的数据：点数不应增加（保持为 1）
        await store.sync(a, [[1.0, 0.0]], 'embedding-v1')
        assert len(db.points) == 1
        
        # 内容发生变更（如法条修订）：生成新语料指纹，点数变为 2
        await store.sync([row('改版条文')], [[0.0, 1.0]], 'embedding-v1')
        assert len(db.points) == 2 and store.corpus_id != old_corpus
        
        # 搜索测试：确保查询时自动加上了新批次的过滤条件，绝不会搜出旧版条文
        hits = await store.search([0.0, 1.0], 12)
        assert len(hits) == 1 and hits[0][0] == 0
        assert db.last_query['filter']['must'][0]['match']['value'] == store.corpus_id
        assert all(p['payload']['source_url'] for p in db.points.values())
        
        await store.close()

    asyncio.run(run())


def test_incompatible_collection_is_not_recreated_or_cleared():
    """测试集合保护机制：当已存在同名集合但向量维度不一致时（如配置为99维），直接报错保护，绝不能静默清空用户原有集合"""
    async def run():
        db = Database()
        db.config = {'vectors': {'size': 99, 'distance': 'Cosine'}}  # 已存在一个99维集合
        store = QdrantStore(transport=httpx.MockTransport(db.handle))
        
        # 尝试写入 2 维向量：必须抛出 RuntimeError 拒绝写入
        with pytest.raises(RuntimeError):
            await store.sync([row()], [[1.0, 0.0]], 'embedding-v1')
            
        # 确认原有集合配置和数据完整未被篡改
        assert db.config['vectors']['size'] == 99 and not db.points
        await store.close()

    asyncio.run(run())



def test_nonlocal_database_endpoint_is_rejected():
    with pytest.raises(ValueError):QdrantStore('http://10.0.0.1:6333')


def test_plain_doc_format_keeps_old_collection_and_corpus_identity():
    """plain 的集合名与批次指纹必须和改造前逐字节一致；新格式两者都变。"""
    rows = [row()]
    model_id = 'embedding-v1'
    old_collection = 'legal_rag_' + hashlib.sha256(model_id.encode()).hexdigest()[:12] + '_2'
    old_corpus = hashlib.sha256(json.dumps([model_id, rows], ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    assert collection_name(model_id, 2) == old_collection
    assert corpus_fingerprint(model_id, rows) == old_corpus
    assert collection_name(model_id, 2, 'law_header_v1') != old_collection
    assert collection_name(model_id, 2, 'law_header_v1').endswith('_law_header_v1_2')
    assert corpus_fingerprint(model_id, rows, 'law_header_v1') != old_corpus


def test_new_doc_format_sync_uses_separate_collection_and_payload_marker():
    """新格式写入独立集合，payload 带格式标记；原有 plain 集合指纹不被复用。"""
    async def run():
        db = Database()
        store = QdrantStore(transport=httpx.MockTransport(db.handle))
        await store.sync([row()], [[1.0, 0.0]], 'embedding-v1', 'law_header_v1')
        assert store.collection.endswith('_law_header_v1_2')
        assert all(p['payload']['doc_format'] == 'law_header_v1' for p in db.points.values())
        await store.sync([row()], [[1.0, 0.0]], 'embedding-v1')
        assert not store.collection.endswith('_law_header_v1_2')
        await store.close()

    asyncio.run(run())


def test_changed_payload_cannot_become_model_evidence():
    async def run():
        db=Database();store=QdrantStore(transport=httpx.MockTransport(db.handle))
        await store.sync([row()],[[1.,0.]],'embedding-v1')
        next(iter(db.points.values()))['payload']['text']='被篡改的内容'
        with pytest.raises(RuntimeError):await store.search([1.,0.],12)
        await store.close()
    asyncio.run(run())


def test_retriever_uses_database_query_and_explicitly_degrades_on_failure(tmp_path):
    from local_rag.retrieval import Retriever
    class Embedding:
        model_id='test-model'
        async def embed(self,texts,*,query=False):return [[1.,0.] for _ in texts]
    async def run():
        db=Database(); store=QdrantStore(transport=httpx.MockTransport(db.handle))
        retriever=Retriever([row()],Embedding(),tmp_path/'index.json',store=store)
        await retriever.build()
        hits,mode=await retriever.search('工伤')
        assert hits and mode=='hybrid_qdrant' and db.last_query
        store.ready=False
        with pytest.raises(RuntimeError, match='尚未完整写入'):
            await store.search([1., 0.], 4)
        hits,mode=await retriever.search('工伤')
        assert hits and mode=='lexical_qdrant_unavailable'
        assert retriever.state=='lexical' and retriever.error
        assert store.ready is False
        await retriever.build()
        assert retriever.state=='ready' and retriever.error==''
        await store.close()
    asyncio.run(run())


@pytest.mark.parametrize('failure_at', ['qdrant', 'embedder', 'local_embedder'])
def test_transient_query_failure_recovers_without_rebuild(tmp_path, failure_at):
    """一次网络故障只能影响本轮；下一轮须恢复混合检索及状态，不能靠重建索引解锁。"""
    from local_rag.retrieval import Retriever

    class Embedding:
        model_id = 'test-model'
        fail_query = False

        async def embed(self, texts, *, query=False):
            if query and self.fail_query:
                self.fail_query = False
                raise TimeoutError('模拟问题向量化超时')
            return [[1., 0.] for _ in texts]

    async def run():
        db, embedding = Database(), Embedding()
        fail_database_query = False

        def handle(request):
            nonlocal fail_database_query
            if request.url.path.endswith('/points/query') and fail_database_query:
                fail_database_query = False
                raise httpx.ReadTimeout('模拟查询超时', request=request)
            return db.handle(request)

        store = None if failure_at == 'local_embedder' else QdrantStore(transport=httpx.MockTransport(handle))
        retriever = Retriever([row()], embedding, tmp_path / 'index.json', store=store)
        try:
            await retriever.build()
            hybrid_mode = 'hybrid_qdrant' if store else 'hybrid'
            baseline, mode = await retriever.search('工伤')
            assert baseline and mode == hybrid_mode
            fail_database_query = failure_at == 'qdrant'
            embedding.fail_query = failure_at != 'qdrant'

            hits, mode = await retriever.search('工伤')
            assert hits and mode == ('lexical_qdrant_unavailable' if store else 'lexical_fallback')
            assert retriever.state == 'lexical' and retriever.error
            if store:
                assert store.ready is True

            # 不再调用 build/sync，也不手动重置 ready；故障替身仅失败一次。
            recovered, mode = await retriever.search('工伤')
            assert recovered == baseline and mode == hybrid_mode
            assert retriever.state == 'ready' and retriever.error == ''
        finally:
            if store:
                await store.close()

    asyncio.run(run())
