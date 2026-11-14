"""模型适配器启动边界：缺少标签或接口异常均不能结束已有Ollama进程。"""
import asyncio

import httpx
import pytest

from local_rag import ollama


def test_adapter_preserves_existing_service_with_missing_models(monkeypatch):
    async def run():
        client=ollama.OllamaClient()
        async def tags():return {'connected':True,'models':[],'chat_ready':False,'embedding_ready':False}
        monkeypatch.setattr(client,'_query_tags',tags)
        monkeypatch.setattr(ollama,'find_ollama_binary',lambda:'ollama.exe')
        monkeypatch.setattr(ollama.subprocess,'run',lambda *a,**k:pytest.fail('不得按名称杀进程'))
        monkeypatch.setattr(ollama.subprocess,'Popen',lambda *a,**k:pytest.fail('已有服务不得重复启动'))
        try:
            status=await client.status()
            assert status['connected'] and not status['chat_ready']
        finally:await client.close()
    asyncio.run(run())


def test_adapter_starts_only_when_port_is_free(monkeypatch):
    async def run():
        client=ollama.OllamaClient()
        commands=[]
        async def tags():
            if not commands:raise httpx.ConnectError('not started')
            return {'connected':True,'models':['local'],'chat_ready':True,'embedding_ready':True}
        def closed_port(*args,**kwargs):raise ConnectionRefusedError()
        async def no_wait(delay):pass
        monkeypatch.setattr(client,'_query_tags',tags)
        monkeypatch.setattr(ollama,'find_ollama_binary',lambda:'ollama.exe')
        monkeypatch.setattr(ollama.socket,'create_connection',closed_port)
        monkeypatch.setattr(ollama.asyncio,'sleep',no_wait)
        monkeypatch.setattr(ollama.subprocess,'Popen',lambda command,**kw:commands.append(command))
        monkeypatch.setattr(ollama.subprocess,'run',lambda *a,**k:pytest.fail('不应杀进程'))
        try:
            status=await client.status()
            assert status['chat_ready']
            assert commands==[['ollama.exe','serve']]
        finally:await client.close()
    asyncio.run(run())


def test_adapter_preserves_listening_service_when_api_fails(monkeypatch):
    async def run():
        client=ollama.OllamaClient()
        async def tags():raise httpx.ConnectError('not ready')
        class Connected:
            def __enter__(self):return self
            def __exit__(self,*args):pass
        monkeypatch.setattr(client,'_query_tags',tags)
        monkeypatch.setattr(ollama,'find_ollama_binary',lambda:'ollama.exe')
        monkeypatch.setattr(ollama.subprocess,'run',lambda *a,**k:pytest.fail('不得按名称杀进程'))
        monkeypatch.setattr(ollama.subprocess,'Popen',lambda *a,**k:pytest.fail('端口占用不得启动新服务'))
        # 补丁后适配器使用TCP探测；旧实现先触发上面的危险命令断言。
        import socket
        monkeypatch.setattr(socket,'create_connection',lambda *a,**k:Connected())
        try:
            status=await client.status()
            assert not status['connected']
        finally:await client.close()
    asyncio.run(run())


def test_embed_applies_query_prefix_only_to_queries(monkeypatch):
    """默认前缀与改造前逐字节一致；空串只供测量脚本关闭前缀；入库文本永不加前缀。"""
    class Reply:
        def raise_for_status(self):pass
        def json(self):return {'embeddings':[[0.0]]}
    class FakeHTTP:
        def __init__(self):self.calls=[]
        async def post(self,path,json):
            self.calls.append(list(json['input']))
            return Reply()
        async def aclose(self):pass

    async def run():
        expected=[(None,ollama.DEFAULT_QUERY_PREFIX+'问题'),('','问题')]
        for prefix,prefixed in expected:
            client=ollama.OllamaClient(query_prefix=prefix)
            fake=FakeHTTP()
            monkeypatch.setattr(client,'http',fake)
            try:
                await client.embed(['问题'],query=True)
                await client.embed(['法条'],query=False)
            finally:
                await client.close()
            assert fake.calls==[[prefixed],['法条']]
    asyncio.run(run())
