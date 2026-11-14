"""本机模型适配：只与本地Ollama通信，并在未启动时自动拉起本机Ollama服务。
embed()用于向量化，chat()用于生成回答；请求失败不会自动转到云端。

阅读指南：docs/guides/PYCHARM_PROJECT_READING.md（相对项目根目录）。
"""
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from urllib.parse import urlsplit

import httpx
from .local_runtime import models_directory

ROOT = Path(__file__).resolve().parents[1]

# 查询侧任务说明（Instruct 前缀）：只加在“用户问题”这一侧，法条入库时不加，因此不需要重建索引。
# 默认值与改造前逐字节一致；本轮只把它变成可配置项，生产默认值不变。
DEFAULT_QUERY_PREFIX = (
    "Instruct: Given a legal question, retrieve relevant Chinese laws and case excerpts.\nQuery: "
)


def resolve_ollama_models_dir():
    """沿用启动器的显式配置优先规则；不搬移模型，已有服务仍以其启动目录为准。"""
    return models_directory(ROOT)


def find_ollama_binary():
    """查找本机安装的 Ollama 可执行文件路径。"""
    ollama = shutil.which("ollama")
    if ollama:
        return ollama
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"
    if candidate.is_file():
        return str(candidate)
    return None


class OllamaClient:
    """本机Ollama适配器：分别负责资料向量化和模型回答，并在服务未就绪时自动启动本机Ollama。"""
    def __init__(self, base_url="http://127.0.0.1:11434", model="qwen3:4b-instruct-2507-q4_K_M", embedding="qwen3-embedding:0.6b",
                 query_prefix=None):
        """初始化对象依赖与内部状态；后续读写通过本类的方法完成。"""
        url = urlsplit(base_url)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "::1", "localhost"}
                or url.username or url.password or url.path not in {"", "/"} or url.query or url.fragment):
            raise ValueError("本地模式仅允许本机HTTP环回服务")
        for tag in (model, embedding):
            if any(x in tag.lower() for x in ("cloud", "http", "/", "\\")):
                raise ValueError("仅允许本机已安装的模型标签，不允许云端模型")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding = embedding
        self.model_id = embedding
        # 空串表示“查询侧不加任何前缀”，只供测量脚本做对照；None 保持生产默认前缀。
        self.query_prefix = DEFAULT_QUERY_PREFIX if query_prefix is None else query_prefix
        self._boot_lock = asyncio.Lock()
        self.http = httpx.AsyncClient(base_url=self.base_url, trust_env=False, follow_redirects=False,
                                     timeout=httpx.Timeout(180, connect=5))

    async def _query_tags(self):
        """请求 /api/tags 并解析模型列表。"""
        r = await self.http.get("/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        self.model_id = self.embedding + ":" + next((x.get("digest", "unknown") for x in models if x["name"] == self.embedding), "missing")
        names = [m["name"] for m in models if "cloud" not in m["name"].lower()]
        return {"connected": True, "models": names, "chat_ready": self.model in names,
                "embedding_ready": self.embedding in names}

    async def _auto_start_ollama(self):
        """只在默认本机端口空闲时拉起Ollama；已有服务缺模型或暂未响应均不重启。

        不按进程名结束服务；目录配置只传给本次新进程，不能改变已有服务的模型目录。
        """
        url = urlsplit(self.base_url)
        if (url.port or 80) != 11434:
            return
        ollama = find_ollama_binary()
        if not ollama:
            return
        async with self._boot_lock:
            try:
                await self._query_tags()
                return  # 接口可用就保留服务；缺模型由状态提示，不接管其他项目。
            except Exception:
                pass
            try:
                with socket.create_connection((url.hostname, 11434), timeout=0.5):
                    return  # 监听者暂未返回JSON也不能视为本项目可终止的进程。
            except OSError:
                pass
            env = dict(
                os.environ,
                OLLAMA_NO_CLOUD="1",
                OLLAMA_HOST=url.netloc,
                OLLAMA_MODELS=resolve_ollama_models_dir(),
            )
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            subprocess.Popen([ollama, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=flags)
            for _ in range(16):
                await asyncio.sleep(0.5)
                try:
                    info = await self._query_tags()
                    if info["chat_ready"]:
                        return
                except Exception:
                    pass

    async def status(self):
        """返回就绪状态、模型/索引信息；若本机Ollama未启动则自动拉起。"""
        try:
            info = await self._query_tags()
            if info["chat_ready"] and info["embedding_ready"]:
                return info
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        await self._auto_start_ollama()
        try:
            return await self._query_tags()
        except (httpx.HTTPError, ValueError, KeyError):
            return {"connected": False, "models": [], "chat_ready": False, "embedding_ready": False}

    async def embed(self, texts, *, query=False):
        """调用本地embedding模型生成向量；查询文本使用检索任务前缀，不把问题写入资料向量缓存。"""
        if query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        r = await self.http.post("/api/embed", json={"model": self.embedding, "input": texts,
                                                   "truncate": False, "keep_alive": "5m"})
        r.raise_for_status()
        return r.json()["embeddings"]

    async def chat(self, messages):
        # No pull, external URL, tools or cloud fallback is available in the consultation path.
        """请求本机问答模型并读取输出；异常、空回答或长度截断明确报错，不转用云端。"""
        ready = await self.status()
        if not ready["chat_ready"]:
            raise RuntimeError("本机问答模型尚未就绪，请先完成模型下载；没有转用云端")
        chunks = []
        async with self.http.stream("POST", "/api/chat", json={
            "model": self.model, "messages": messages, "stream": True, "think": False,
            "keep_alive": "5m", "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 650},
        }) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError("本机模型推理失败，请检查Ollama状态")
                if data.get("done_reason") == "length":
                    raise RuntimeError("模型回答达到长度限制，未作为完整答案展示，请缩短问题后重试")
                chunks.append(data.get("message", {}).get("content", ""))
                if sum(map(len, chunks)) > 20000:
                    raise RuntimeError("模型输出过长，已停止")
        answer = "".join(chunks).strip()
        if not answer:
            raise RuntimeError("本机模型返回空回答")
        return answer

    async def close(self):
        """释放HTTP客户端连接；不删除模型、资料或数据库集合。"""
        await self.http.aclose()
