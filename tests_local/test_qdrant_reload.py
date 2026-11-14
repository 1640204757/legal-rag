"""Web 前端“加载新资料并入库”接口热重载自动化测试

用途说明：
验证用户在网页端点击【加载新资料并入库】（/api/index/rebuild 接口）时，服务能否在【无需重启 Web 服务】的前提下：
1. 自动重新读取磁盘新抓取的法条数据；
2. 增量计算新法条的向量；
3. 将新批次安全同步写入 Qdrant；
4. 自动更新网页端显示的条目数与批次指纹。

测试特点：
全程使用内存虚拟 Model 与虚拟 Database，不修改用户真实的本地向量库。
"""
import time
import httpx
from fastapi.testclient import TestClient
import local_rag.app as app_module
import local_rag.retrieval as retrieval_module
from local_rag.qdrant import QdrantStore
from test_qdrant import Database


def test_rebuild_loads_new_corpus_without_service_restart(monkeypatch, tmp_path):
    """测试热重载接口：初始有1条法条，向语料库追加第2条法条后触发 /api/index/rebuild，验证服务无缝更新"""
    # 模拟初始资料只有 1 条
    rows = [{'id': 'doc/a', 'text': '条文甲', 'title': '测试规范'}]

    class Model:
        """测试用快速模型桩：即时返回虚拟向量，支持状态查询与异步关闭"""
        model_id = 'reload-test-model'
        def __init__(self, *args):
            self.model = 'test-chat'
            self.embedding = 'test-embedding'
        async def status(self):
            return {'connected': True, 'chat_ready': True, 'embedding_ready': True}
        async def embed(self, texts, *, query=False):
            return [[1.0, 0.0] for _ in texts]
        async def close(self):
            pass

    def corpus():
        """模拟读取磁盘语料库的动态函数"""
        return [dict(r) for r in rows], {'laws': 1, 'cases': 0, 'chunks': len(rows)}

    db = Database()
    # 通过 monkeypatch 劫持底层网络与模型，使其在纯内存环境中运行
    monkeypatch.setattr(app_module, 'OllamaClient', Model)
    monkeypatch.setattr(app_module, 'QdrantStore', lambda url: QdrantStore(transport=httpx.MockTransport(db.handle)))
    monkeypatch.setattr(app_module, 'load_corpus', corpus)
    monkeypatch.setattr(retrieval_module, 'ROOT', tmp_path)

    def wait_ready(client, total):
        """轮询接口，等待后台异步索引任务完成"""
        for _ in range(60):
            s = client.get('/api/status').json()
            if s['index']['state'] == 'ready' and s['index']['total'] == total:
                return s
            time.sleep(0.01)
        raise AssertionError('索引没有完成重载')

    with TestClient(app_module.create_app()) as client:
        # 1. 验证启动时第一批数据（1条）加载完成
        first = wait_ready(client, 1)
        
        # 2. 模拟外部采集脚本向磁盘写入了第 2 条新法条
        rows.append({'id': 'doc/b', 'text': '新增条文乙', 'title': '测试规范'})
        
        # 3. 模拟用户在网页前端点击【加载新资料并入库】按钮
        response = client.post('/api/index/rebuild', headers={'x-local-token': first['token']})
        assert response.status_code == 200
        
        # 4. 验证服务在不重启的情况下，总条目数和数据库点数自动变为 2，并生成新批次指纹
        updated = wait_ready(client, 2)
        assert updated['coverage']['chunks'] == 2
        assert updated['database']['points'] == 2
        assert updated['database']['corpus_id'] != first['database']['corpus_id']

