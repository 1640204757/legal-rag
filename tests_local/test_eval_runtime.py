"""观测器单元测试使用明确标记的HTTP桩，不计入真实40题得分。"""
import asyncio
import httpx
import gzip
from local_rag.eval_runtime import RecordingTransport, materials_in_prompt,stream_answer
from prototype.rag.input_builder import Material


def test_transport_records_actual_response_bytes_and_rejects_qdrant_writes():
    async def run():
        events=[]
        transport=RecordingTransport(events,'qdrant',inner=httpx.MockTransport(lambda request:httpx.Response(200,json={'result':{'points':[]}})))
        async with httpx.AsyncClient(transport=transport) as client:
            response=await client.post('http://127.0.0.1:16333/collections/a/points/query',json={'query':[1.0]})
            assert response.json()['result']['points']==[]
            assert events[0]['response_json']['result']['points']==[]
            assert events[0]['complete'] and events[0]['elapsed_seconds']>=0
            try:
                await client.put('http://127.0.0.1:16333/collections/a',json={})
            except RuntimeError:
                pass
            else:
                raise AssertionError('评测不得写Qdrant')
    asyncio.run(run())


def test_observation_requires_complete_rendered_material():
    row=dict(kind='LAW_ARTICLE',title='测试法',locator='第一条',version_id='v1',text='条文全文',
             source_url='https://example.test',verified_scope='快照',content_scope='ARTICLES_ONLY')
    material=Material(material_id='M1',**row)
    assert materials_in_prompt([row],material.render())[0]['text']=='条文全文'
    assert materials_in_prompt([row],material.render()[:-4])==[]


def test_single_packet_and_truncated_stream_are_preserved():
    packet=dict(message=dict(content='实际输出'),done=True,done_reason='stop')
    text,reasons=stream_answer([dict(service='ollama',url='http://127.0.0.1:11434/api/chat',response_json=packet)])
    assert text=='实际输出' and reasons==['stop']
    text,reasons=stream_answer([dict(service='ollama',url='http://127.0.0.1:11434/api/chat',response_text='{"message":{"content":"部分输出"}}\n{"message":')])
    assert text=='部分输出' and reasons==[]


def test_gzip_qdrant_response_keeps_payload_and_original_bytes():
    async def run():
        events=[]
        raw=gzip.compress(b'{"result":{"points":[{"payload":{"title":"test"}}]}}')
        inner=httpx.MockTransport(lambda request:httpx.Response(200,content=raw,headers={'Content-Encoding':'gzip'}))
        async with httpx.AsyncClient(transport=RecordingTransport(events,'qdrant',inner=inner)) as client:
            response=await client.post('http://127.0.0.1:16333/a/points/query',json={})
            assert response.json()['result']['points'][0]['payload']['title']=='test'
            assert events[0]['response_json']['result']['points'][0]['payload']['title']=='test'
            import base64
            assert base64.b64decode(events[0]['response_bytes_base64'])==raw
    asyncio.run(run())
