"""网页API入口：接收问题、创建会话、查询任务状态、停止与清空会话。
create_app()负责组装服务；业务流程交给service.py，资料检索交给retrieval.py。

阅读指南：docs/guides/PYCHARM_PROJECT_READING.md（相对项目根目录）。
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import os
import mimetypes
import secrets

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from .ollama import OllamaClient
from .retrieval import Retriever, load_corpus
from .service import ChatService, SessionError
from .qdrant import QdrantStore
from .official_capture import fetch_official_npc_slice
from .local_runtime import APP_ID, workspace_identity

HERE = Path(__file__).parent
# Windows registry may associate .js with text/plain; nosniff then blocks the app.
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


class Message(BaseModel):
    """提交问题的数据格式：限制会话编号和问题长度，防止空输入或过大请求。"""
    session_id: str = Field(min_length=20, max_length=100)
    question: str = Field(min_length=1, max_length=2000)


class SessionRequest(BaseModel):
    """仅带会话编号的请求格式，供会话清空等接口复用。"""
    session_id: str = Field(min_length=20, max_length=100)


class CancelRequest(SessionRequest):
    """携带会话及任务编号，用于查询或停止对应任务。"""
    request_id: str = Field(min_length=10, max_length=100)


def create_app(service=None):
    """组装应用及生命周期，默认本机模型/Qdrant与受控国家库兜底；注入service时不改其配置。

    启动会读取本地语料，兜底仅在问答缺失专项时调用；真实采集只写独立缓存，不发布法规。
    """
    token = secrets.token_urlsafe(32)
    identity = dict(app_id=APP_ID, workspace_id=workspace_identity(HERE.parent), process_id=os.getpid())
    state = {"service": service, "coverage": {}, "index_task": None}

    async def prepare_index():
        # 重新读取磁盘资料，让新增已核验条文无需改接口或重启服务即可入库。
        """重新加载磁盘资料并准备向量入库；保留已生成向量缓存，只有整批核对通过才启用Qdrant查询。"""
        retriever = state['retriever']
        try:
            rows,coverage = load_corpus()
        except Exception:
            retriever.error = '新资料校验失败，未替换当前已加载资料；请检查采集记录'
            return
        retriever.rows=rows
        retriever.vectors=None
        retriever.store.ready=False
        state['coverage']=coverage
        model = state["model"]
        status = await model.status()
        if status["embedding_ready"]:
            await state["retriever"].build()
        else:
            state["retriever"].state = "lexical"
            state["retriever"].error = "向量模型未安装，当前仅关键词检索"

    @asynccontextmanager
    async def lifespan(app):
        """在服务启动时创建模型、数据库和后台任务；关闭时取消任务并释放连接和内存会话。"""
        if state["service"] is None:
            model = OllamaClient(os.getenv("OLLAMA_LOCAL_URL", "http://127.0.0.1:11434"),
                                 os.getenv("LOCAL_CHAT_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
                                 os.getenv("LOCAL_EMBED_MODEL", "qwen3-embedding:0.6b"))
            rows, coverage = load_corpus()
            store = QdrantStore(os.getenv('QDRANT_LOCAL_URL','http://127.0.0.1:16333'))
            retriever = Retriever(rows, model, store=store)
            state.update(model=model, store=store, retriever=retriever, coverage=coverage,
                         service=ChatService(retriever, model, official_fetcher=fetch_official_npc_slice))
            state["index_task"] = asyncio.create_task(prepare_index())
        async def clean():
            """周期检查并清理闲置过期会话；这是后台任务，退出服务时会被取消。"""
            while True:
                await asyncio.sleep(30)
                state["service"].cleanup()
        cleaner = asyncio.create_task(clean())
        try:
            yield
        finally:
            cleaner.cancel()
            jobs = [cleaner]
            if state["index_task"]:
                state["index_task"].cancel()
                jobs.append(state["index_task"])
            svc = state["service"]
            for session in svc.sessions.values():
                session.valid = False
                for job in session.jobs.values():
                    job["task"].cancel()
                    jobs.append(job["task"])
            await asyncio.gather(*jobs, return_exceptions=True)
            if "model" in state:
                await state["model"].close()
                await state['store'].close()
            svc.sessions.clear()

    app = FastAPI(title="OpenLawGuide 本地版", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        """限制修改请求的来源和本机令牌，并设置页面安全响应头；不将咨询发送到外部站点。"""
        if request.method not in {"GET", "HEAD"}:
            origin = request.headers.get("origin")
            own_origin = str(request.base_url).rstrip("/")
            if ((origin and origin != own_origin)
                    or not secrets.compare_digest(request.headers.get("x-local-token", ""), token)):
                return JSONResponse({"error": "仅允许本机页面操作，请刷新页面重试"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.exception_handler(SessionError)
    async def session_error(request, exc):
        """把会话错误转换为明确HTTP状态与可读提示，不暴露内部异常细节。"""
        return JSONResponse({"error": str(exc)}, status_code=exc.code)

    @app.get("/")
    async def home():
        """返回本地HTML首页；样式和脚本由同源静态目录提供。"""
        return FileResponse(HERE / "static/index.html")

    @app.get("/api/status")
    async def status():
        """返回就绪状态、模型/索引信息；若模型刚自动启动完成，自动触发向量索引与Qdrant连接。"""
        if "model" not in state:
            return {**identity, "token": token, "local_only": True, "coverage": {}, "model": "test", "connected": True,
                    "chat_ready": True, "index": {"state": "lexical", "completed": 0, "total": 0}}
        result = await state["model"].status()
        r = state["retriever"]
        if (result.get("embedding_ready") and r.state != "ready"
                and r.error == "向量模型未安装，当前仅关键词检索"
                and (state["index_task"] is None or state["index_task"].done())):
            state["index_task"] = asyncio.create_task(prepare_index())
        return {**result, **identity, "token": token, "local_only": True, "coverage": state["coverage"],
                "model": state["model"].model, "embedding": state["model"].embedding,
                "index": {"state": r.state, "completed": r.completed, "total": len(r.rows), "notice": r.error},
                'database':state['store'].status()}

    @app.post("/api/session")
    async def session():
        """创建新的内存会话，返回随机会话编号；不创建聊天磁盘文件。"""
        return {"session_id": state["service"].create()}

    @app.post("/api/chat/start")
    async def start(message: Message):
        """校验并启动本轮异步问答，返回任务编号；实际计算由后台流程完成。"""
        question = message.question.strip()
        if not question:
            return JSONResponse({"error": "请先输入问题"}, status_code=422)
        return {"request_id": state["service"].start(message.session_id, question)}

    @app.post("/api/chat/result")
    async def result(message: CancelRequest):
        """读取指定会话中的任务状态与结果，不能跨会话读取其他任务。"""
        return state["service"].result(message.session_id, message.request_id)

    @app.post("/api/chat/cancel")
    async def cancel(message: CancelRequest):
        """请求取消指定任务并等待清理；不把未完成回答写入聊天历史。"""
        await state["service"].cancel(message.session_id, message.request_id)
        return {"status": "cancelled"}

    @app.post("/api/chat/reset")
    async def reset(message: SessionRequest):
        """清空旧会话并创建新会话；先作废旧对象，防止在途任务完成后晚写回。"""
        return {"session_id": await state["service"].reset(message.session_id)}

    @app.post("/api/index/rebuild")
    async def rebuild():
        """在没有问答运行时安排资料重载和向量入库；重复点击不会并发创建多个构建任务。"""
        task = state["index_task"]
        if task and not task.done():
            return {"state": "building"}
        if "model" not in state:
            return JSONResponse({"error": "模型未连接"}, status_code=503)
        svc = state["service"]
        if any(j["status"] == "running" for s in svc.sessions.values() for j in s.jobs.values()):
            return JSONResponse({"error": "正在分析问题，请结束后再重建"}, status_code=409)
        state["index_task"] = asyncio.create_task(prepare_index())
        return {"state": "building"}

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    return app
