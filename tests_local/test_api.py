"""FastAPI 本地接口与安全防御自动化测试

用途说明：
本文件使用 FastAPI 的 TestClient 模拟浏览器发起 HTTP 请求，验证本地 Web 服务的安全防御与会话隔离机制。
为了保证测试速度和独立性，这里使用了测试替身（Mock），不需要真实启动 Ollama 或 Qdrant。

核心测试覆盖点：
1. 本地安全边界（防外网跨域恶意调用、CSRF Token 校验、CSP 安全响应头）。
2. 会话隔离（A 用户的 session_id 绝不能读取 B 用户的异步任务结果）。
3. 虚假法条引用拦截（模型虚构 M99 等不存在的依据编号时，回答必须被拦截）。
4. 超范围问题安全兜底（未覆盖的法律领域不能拿无关法条胡乱作答）。
5. 虚构法定期限拦截（条文中没有写时限而模型瞎编“1年内”时，必须被拦截）。
"""
import asyncio
from fastapi.testclient import TestClient
from local_rag.app import create_app
from local_rag.service import ChatService


class Search:
    """测试用检索桩（Mock Retriever）：模拟检索模块，仅返回空列表，避免测试依赖真实数据库"""
    async def search(self, query, top_k=4):
        return [], "lexical"


class Model:
    """测试用大模型桩（Mock LLM）：模拟本机大模型，返回标准格式的预设文本，避免测试调用真实模型"""
    async def chat(self, messages):
        from tests_local.test_answer_permission import reading
        return reading(quote=None)


def test_local_api_origin_reset_and_session_isolation():
    """测试接口来源校验、Token 防御、跨会话越权防御与会话重置
    
    验证步骤：
    1. 首页与静态资源加载正常（HTTP 200，正确的 Content-Type）。
    2. 安全头包含 Content-Security-Policy。
    3. 未携带 x-local-token 时拒绝 POST 请求（HTTP 403）。
    4. 携带恶意来源 Origin（如第三方钓鱼网站）时坚决拒绝（HTTP 403）。
    5. 能够正常创建会话并提交问题，返回合法任务编号 request_id。
    6. 用另一个会话 ID 试图读取其他用户的任务结果时，返回 404（实现会话严格隔离）。
    7. 调用重置接口后，原有会话 ID 立即作废（再次提问返回 410）。
    """
    service = ChatService(Search(), Model())
    with TestClient(create_app(service)) as c:
        # 验证前端静态页面正常访问
        assert c.get('/').status_code == 200
        assert 'javascript' in c.get('/static/app.js').headers['content-type']
        assert 'Content-Security-Policy' in c.get('/').headers
        
        # 无 Token 访问接口应被拦截
        assert c.post('/api/session').status_code == 403
        
        # 获取合法本地 Token
        status = c.get('/api/status').json()
        from local_rag.local_runtime import APP_ID, workspace_identity
        from local_rag.app import HERE
        import os
        assert status['app_id'] == APP_ID
        assert status['workspace_id'] == workspace_identity(HERE.parent)
        assert status['process_id'] == os.getpid()
        assert str(HERE.parent) not in str({k:status[k] for k in ('app_id','workspace_id','process_id')})
        token = status['token']
        h = {'x-local-token': token}
        
        # 防跨站攻击测试：非本地 Origin 必须被拒绝
        assert c.post('/api/session', headers={**h, 'origin': 'https://evil.example'}).status_code == 403
        
        # 正常创建会话 A 与 会话 B
        sid = c.post('/api/session', headers=h).json()['session_id']
        other = c.post('/api/session', headers=h).json()['session_id']
        
        # 在会话 A 中启动一次问答任务
        started = c.post('/api/chat/start', headers=h, json={'session_id': sid, 'question': '工资问题'})
        assert started.status_code == 200
        rid = started.json()['request_id']
        
        # 关键安全测试：用会话 B 的身份去查会话 A 的结果，必须返回 404（防止多标签页或用户串号泄露）
        assert c.post('/api/chat/result', headers=h, json={'session_id': other, 'request_id': rid}).status_code == 404
        
        # 会话重置测试：重置后生成全新 session_id，旧 session_id 彻底失效
        new = c.post('/api/chat/reset', headers=h, json={'session_id': sid}).json()['session_id']
        assert new != sid
        assert c.post('/api/chat/start', headers=h, json={'session_id': sid, 'question': '旧会话'}).status_code == 410


def test_unknown_model_citation_not_committed():
    """测试虚假法条引用拦截：当大模型捏造未提供的材料编号（如 M99）时，系统绝不采信该回答"""
    class Unsafe:
        """恶意或幻觉模型替身：故意在回答中引用不存在的 M99 编号"""
        async def chat(self, messages):
            return '## 处理方向\n依据 M99 可以拿到赔偿。\nUSED_IDS: M99'

    async def run():
        svc = ChatService(Search(), Unsafe())
        sid = svc.create()
        rid = svc.start(sid, '测试')
        await asyncio.sleep(0.02)
        result = svc.result(sid, rid)['result']
        
        # 核心断言：检测到非法引用编号时，必须判定 blocked=True 并清空引用卡片
        assert result['blocked'] and result['citations'] == []
        assert result['raw_answer'].endswith('M99')
        # 被拦截的错误回答绝不写入会话历史
        assert svc.sessions[sid].history == []

    asyncio.run(run())


def test_corpus_unsupported_domain_never_supplied_to_model():
    """测试知识库超出领域拦截：对于未收录的法律领域（如问刑事判刑），不得混用其他领域法条误导模型"""
    class ShouldNotSearch:
        """严格检索桩：如果检索逻辑错误地去查了无关资料，直接抛出断言错误"""
        async def search(self, query, top_k=4):
            raise AssertionError('不应检索无关资料来回答刑事问题')

    class Remember:
        """检查输入消息的桩：验证系统是否向模型明确注入了“所属领域尚无知识资料”的警告提示"""
        async def chat(self, messages):
            assert '该问题所属领域尚无知识资料' in str(messages)
            return '## 不确定性\n未覆盖。\nUSED_IDS: 无'

    async def run():
        svc = ChatService(ShouldNotSearch(), Remember())
        sid = svc.create()
        rid = svc.start(sid, '诈骗怎么判刑')
        await asyncio.sleep(0.02)
        result = svc.result(sid, rid)
        assert result['status'] == 'done'
        # 确认确实没有错误召回无关材料
        assert not result['result']['retrieved']

    asyncio.run(run())


def test_real_service_blocks_deadline_even_when_material_id_exists():
    """测试法定期限虚构拦截：法条编号虽然合法（M1），但如果模型在回答里捏造了法条中不存在的期限，依然必须拦截"""
    class Materials:
        """模拟提供真实法条，但法条里只有提交材料的要求，没有任何关于“1年内”的时限规定"""
        async def search(self, query):
            return [dict(
                kind='LAW_ARTICLE',
                title='测试规范',
                locator='第十八条',
                version_id='未核实',
                text='申请时应提交申请表和诊断证明。',  # 注意：原文完全没有写时效期限
                source_url='https://www.court.gov.cn/',
                verified_scope='测试材料',
                content_scope='ARTICLES_ONLY'
            )], 'lexical'

    class Unsafe:
        """幻觉模型替身：即便引用了合法的 M1，但擅自编造了“1年内”的时限结论"""
        async def chat(self, messages):
            return '## 处理方向\n职工自事故发生之日起1年内申请，依据M1。\nUSED_IDS: M1'

    async def run():
        svc = ChatService(Materials(), Unsafe())
        sid = svc.create()
        rid = svc.start(sid, '准备什么材料？')
        await asyncio.sleep(0.02)
        result = svc.result(sid, rid)['result']
        
        # 核心断言：期限校对拦截器必须精准识别出 DEADLINE_VALUE_NOT_IN_CITED_TEXT 警告并予以拦截
        assert result['blocked']
        assert any(i['type'] == 'DEADLINE_VALUE_NOT_IN_CITED_TEXT' for i in result['warnings'])
        assert svc.sessions[sid].history == []

    asyncio.run(run())

