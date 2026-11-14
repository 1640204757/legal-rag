"""真实评测调度：只读接入已入库语料，观测生产ChatService，不替换其回答或防御逻辑。

入口run_evaluation：初始化一次缓存及Qdrant，逐题真实推理并即时保存网络字节和三层结果。
本模块会访问127.0.0.1:16333/11434并写评测证据；禁止Qdrant写请求，不重算资料向量。
独立防御题明确标记为确定性注入，不混入40题真实模型结果。
"""
import asyncio
import base64
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time
import zlib

import httpx

from .catalog_ingest import _save_json
from .evaluation import verify_frozen,score_hits,audit_citations,judge_degrade,aggregate,article_key
from .ollama import OllamaClient
from .qdrant import QdrantStore
from .retrieval import Retriever,load_corpus,ROOT
from .service import ChatService,deadline_grounding_issues
from prototype.rag.input_builder import Material
from prototype.rag.postprocess import postprocess


class RecordingStream(httpx.AsyncByteStream):
    """原样透传HTTP响应字节并留证，包括推理截断或取消前已收到的内容。"""
    def __init__(self,stream,event):
        self.stream,self.event=stream,event
        self.parts=[]
        self.started=time.perf_counter()

    async def __aiter__(self):
        """只观测，不改响应；正常读完与异常中止分别记录。"""
        try:
            async for chunk in self.stream:
                self.parts.append(chunk)
                yield chunk
            self.event['complete']=True
        except BaseException as exc:
            self.event['stream_error']=type(exc).__name__
            raise
        finally:
            self.save()

    def save(self):
        """解码已收到的JSON或NDJSON原文；流不完整时仍保留部分内容。"""
        raw=b''.join(self.parts)
        self.event['response_bytes_base64']=base64.b64encode(raw).decode('ascii')
        self.event['response_sha256']=hashlib.sha256(raw).hexdigest()
        try:
            encoding=self.event.get('content_encoding','identity').lower()
            decoded=gzip.decompress(raw) if encoding=='gzip' else zlib.decompress(raw) if encoding=='deflate' else raw
            text=decoded.decode('utf-8',errors='strict')
        except (OSError,EOFError,zlib.error,UnicodeError) as exc:
            # 压缩流不完整也保留原字节，不能让观测器解码错误影响真实生产响应。
            self.event['decode_error']=type(exc).__name__
            text=''
        self.event['elapsed_seconds']=time.perf_counter()-self.started+self.event.get('headers_elapsed_seconds',0)
        try:
            self.event['response_json']=json.loads(text)
        except ValueError:
            self.event['response_text']=text

    async def aclose(self):
        """关闭原始连接并保存取消前内容，不将未读完流标成完整。"""
        self.save()
        await self.stream.aclose()


class RecordingTransport(httpx.AsyncBaseTransport):
    """记录真实请求/响应的传输装饰器；不保存请求头，避免泄露数据库认证信息。"""
    def __init__(self,events,service,inner=None):
        self.events,self.service=events,service
        self.inner=inner or httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self,request):
        """真实转发本机只读/推理请求，写库方法在到达网络前拒绝。"""
        if self.service=='qdrant' and not (request.method=='GET' or request.method=='POST' and request.url.path.endswith(('/count','/query','/scroll'))):
            raise RuntimeError('评测传输层禁止写入Qdrant')
        started=time.perf_counter()
        event=dict(service=self.service,method=request.method,url=str(request.url),
                   started_at=datetime.now().astimezone().isoformat(),complete=False)
        try:
            event['request_json']=json.loads(request.content) if request.content else None
        except (ValueError,httpx.RequestNotRead):
            event['request_json']=None
        self.events.append(event)
        try:
            response=await self.inner.handle_async_request(request)
            event['status_code']=response.status_code
            event['content_encoding']=response.headers.get('content-encoding','identity')
            event['headers_elapsed_seconds']=time.perf_counter()-started
            # 构造新的流式响应，确保底层已缓冲响应也会经过观测器，不丢失实际响应字节。
            return httpx.Response(response.status_code,headers=response.headers,
                                  stream=RecordingStream(response.stream,event),extensions=response.extensions)
        except BaseException as exc:
            event['error']=type(exc).__name__
            event['elapsed_seconds']=time.perf_counter()-started
            raise

    async def aclose(self):
        """释放底层HTTP连接池；不关闭本机服务。"""
        await self.inner.aclose()


def materials_in_prompt(hits,prompt):
    """只把完整render字符串确实出现在真实输入中的材料计为已送入，不猜测截断材料。

    候选编号按生产第一轮整条预算重建，最后以真实请求文本逐字确认，不改变实际选择。
    """
    candidates=[]
    count=0
    for row in hits:
        if count+len(row['text'])<=3600:
            count+=len(row['text'])
            names=('kind','title','locator','version_id','text','source_url','verified_scope','content_scope')
            data={name:row[name] for name in names}
            material=Material(material_id=f'M{len(candidates)+1}',**data)
            candidates.append(material)
    return [m.__dict__ for m in candidates if m.render() in prompt]


class ObservedRetriever:
    """代理真实检索，只记录返回值和耗时，不缓存问题结果或替换Top4。"""
    def __init__(self,retriever):
        self.inner=retriever
        self.rows=retriever.rows
        self.hits=[]
        self.elapsed=0
        self.mode=None

    async def search(self,query,top_k=4):
        """透传生产检索默认Top4；同一调用结果同时供服务和评分使用。"""
        started=time.perf_counter()
        self.hits,self.mode=await self.inner.search(query,top_k=top_k)
        self.elapsed=time.perf_counter()-started
        return self.hits,self.mode


class ObservedModel(OllamaClient):
    """使用生产OllamaClient.chat原样推理，仅保留真实送入的消息。"""
    async def chat(self,messages):
        """记录输入后调用父实现，保持原模型、温度、上下文和650 token上限不变。"""
        self.last_messages=messages
        return await super().chat(messages)


def stream_answer(events):
    """从实际Ollama响应流恢复已生成文本及终止原因，保留截断和部分输出。"""
    texts=[]
    reasons=[]
    for event in events:
        if event['service']=='ollama' and event['url'].endswith('/api/chat'):
            packets=[]
            if isinstance(event.get('response_json'),dict):
                packets.append(event['response_json'])
            for line in event.get('response_text','').splitlines():
                try:
                    packets.append(json.loads(line))
                except ValueError:
                    continue
            for packet in packets:
                texts.append(packet.get('message',{}).get('content',''))
                if packet.get('done'):
                    reasons.append(packet.get('done_reason','done'))
    return ''.join(texts),reasons


def expected_row(expected,rows):
    """核验冻结标注的实际条文与文本哈希，语料变动时拒绝偷偷重绑版本。"""
    matches=[r for r in rows if r['title']==expected['title'] and r['version_id']==expected['version_id'] and article_key(r['locator'])==article_key(expected['article_no'])]
    if len(matches)!=1 or hashlib.sha256(matches[0]['text'].encode()).hexdigest()!=expected['text_sha256']:
        raise ValueError('冻结标注与当前语料不一致：'+expected['title']+expected['article_no'])
    return matches[0]


def run_defenses(spec,rows):
    """注入预设输出到生产后处理函数，独立记录实际拦截及评测器识别结果；不调用假模型。"""
    results=[]
    for test in spec['defense_tests']:
        materials=[dict(expected_row(e,rows),material_id=f'M{i+1}') for i,e in enumerate(test['materials'])]
        answer=test['answer']
        checked=postprocess(answer,materials)
        deadline=deadline_grounding_issues(answer,materials)
        actual=checked['blocking'] or bool(deadline)
        results.append(dict(test,mode='deterministic_injection_no_model_call',actual_block=actual,
                            correct=actual==test['expect_block'],production_issues=checked['blocking_issues']+deadline,
                            independent_citation_audit=audit_citations(answer,materials,rows),provided_materials=materials))
    return results


def validate_corpus_scope(spec, corpus_id, coverage, evidence=None):
    """冻结旧corpus身份不改；新库必须匹配已完成扩充凭据，不能任意重绑。"""
    if corpus_id == spec['corpus_id']:
        return 'frozen_baseline'
    if (not evidence or not evidence.get('final') or evidence.get('pending_titles')
            or evidence.get('worker_running') or evidence.get('latest_verified_coverage') != coverage):
        raise ValueError('新语料缺少已完成扩充凭据或覆盖数量不一致')
    latest = evidence.get('live_database_counts', {}).get('latest', {})
    if latest.get('corpus_id') != corpus_id or latest.get('exact_count') != coverage['chunks']:
        raise ValueError('当前语料指纹与已验收新库不一致')
    return 'verified_expansion'


def token_usage(events):
    """只读Ollama真实终止包的Token计数；未返回时用None，不以字数估算。"""
    packets = []
    for event in events:
        if event['service'] != 'ollama' or not event['url'].endswith('/api/chat'):
            continue
        if isinstance(event.get('response_json'), dict):
            packets.append(event['response_json'])
        for line in event.get('response_text', '').splitlines():
            try:
                packets.append(json.loads(line))
            except ValueError:
                continue
    done = [p for p in packets if p.get('done')]
    return dict(prompt_tokens=sum(p['prompt_eval_count'] for p in done) if done and all('prompt_eval_count' in p for p in done) else None,
                completion_tokens=sum(p['eval_count'] for p in done) if done and all('eval_count' in p for p in done) else None,
                completed_response_packets=len(done))


async def initialize(model,store,spec):
    """只读校验真实模型指纹、缓存和当前corpus精确点数；禁止隐式重新入库。"""
    ready=await model.status()
    if not ready['chat_ready'] or not ready['embedding_ready']:
        raise RuntimeError('真实本机问答及向量模型必须同时就绪')
    rows,coverage=load_corpus()
    corpus_id=hashlib.sha256(json.dumps([model.model_id,rows],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    evidence_path=ROOT/'artifacts/local-rag/expansion-20261021/summary.json'
    evidence=json.loads(evidence_path.read_text(encoding='utf-8')) if evidence_path.exists() else None
    judicial_path=ROOT/'data/catalogs/judicial-core-20261021/local-batch-003/completion.json'
    if judicial_path.exists():
        receipt=json.loads(judicial_path.read_text(encoding='utf-8'))
        if receipt.get('corpus_id')==corpus_id:
            # 仅增加新语料准入凭据，不改冻结题集、评分器及生产问答链。
            from .eval_judicial_evidence import normalize_completion
            evidence=normalize_completion(receipt,corpus_id,coverage)
            evidence_path=judicial_path
    scope=validate_corpus_scope(spec,corpus_id,coverage,evidence)
    if (ROOT/'artifacts/local-rag/verified-batch-worker.lock').exists():
        raise ValueError('入库仍在运行，拒绝对变化中的语料评分')
    store.collection='legal_rag_'+hashlib.sha256(model.model_id.encode()).hexdigest()[:12]+'_1024'
    store.corpus_id=corpus_id
    counted=await store.request('POST','/collections/'+store.collection+'/points/count',dict(exact=True,filter=store.scope()))
    if counted['count']!=len(rows):
        raise RuntimeError('当前Qdrant corpus精确计数不符，评测未重建数据库')
    cache_path=ROOT/'data/local_index/embeddings.json'
    cache=json.loads(cache_path.read_text(encoding='utf-8'))
    keys=[hashlib.sha256((r['id']+'\n'+r['text']).encode()).hexdigest() for r in rows]
    fingerprint=hashlib.sha256(json.dumps([model.model_id,keys]).encode()).hexdigest()
    if cache['model_id']!=model.model_id or cache['fingerprint']!=fingerprint:
        raise ValueError('缓存模型或内容指纹不匹配，禁止重算后冒充相同基线')
    vectors=[Retriever.normalize(cache['entries'][key]) for key in keys]
    if {len(v) for v in vectors}!={1024}:
        raise ValueError('缓存维数不符')
    del cache
    # 真实sync先后精确计数相等时只校验；传输层同时禁止任何PUT写入。
    await store.sync(rows,vectors,model.model_id)
    retriever=Retriever(rows,model,store=store)
    retriever.vectors=vectors
    retriever.state='ready'
    retriever.completed=len(rows)
    retriever.evaluation_scope=dict(mode=scope, frozen_corpus_id=spec['corpus_id'], actual_corpus_id=corpus_id,
                                    evidence_file=str(evidence_path) if scope=='verified_expansion' else None,
                                    evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest() if scope=='verified_expansion' else None)
    for question in spec['questions']:
        for expected in question['expect_hit_articles']:
            expected_row(expected,rows)
    return retriever,coverage


def render_markdown(report):
    """把真实分子分母、未通过门槛和逐题失败写为中文报告，不生成虚构合格结论。"""
    summary=report['summary']
    def value(metric):
        return f"{metric['numerator']}/{metric['denominator']}（{metric['rate']:.1%}）" if metric['rate'] is not None else '未取得样本，不能判定'
    lines=['# 评测与验收结果','',f"结论：**{summary['verdict']}**。实际运行：{report['started_at']} 至 {report['finished_at']}。",'',
           f"corpus_id：`{report['corpus_id']}`；语料{report['coverage']['laws']}部规范、{report['coverage']['chunks']:,}段，另有{report['coverage']['cases']}个案例发布稿摘要。",'',
           f"冻结评测SHA256：`{report['benchmark_sha256']}`。40题人工编写咨询场景，逐题真实检索及模型调用，未按得分改题。",'',
           '## 三层指标','',f"- 完整执行：{summary['execution']['completed']}/40；超时、截断和异常不会被排除或视为安全降级。",
           '- 检索 Doc_Hit@4：'+value(summary['retrieval']['doc_hit_at4'])+'；目标≥85%。',
           '- 检索 Article_Hit@4 任一/全部：'+value(summary['retrieval']['article_any_at4'])+' / '+value(summary['retrieval']['article_all_at4'])+'。',
           '- 材料组装后 Doc_Hit / Article任一 / Article全部：'+ ' / '.join(value(summary['assembly'][k]) for k in ('doc_hit_at4','article_any_at4','article_all_at4'))+'。',
           '- 原始输出错误引用率：'+value(summary['citations']['raw_false_rate'])+'。',
           '- 最终展示错误引用率：'+value(summary['citations']['final_false_rate'])+'；目标0%。',
           f"- 待核验显式引用：原始{summary['citations']['raw_unverifiable']}题、最终{summary['citations']['final_unverifiable']}题；不当作通过。",
           '- 范围降级正确：'+value(summary['boundary'])+'。','',
           '## 独立确定性防御','',
           '- 非法期限拦截：'+value(summary['deadline']['interception'])+'；目标100%。',
           '- 非法期限漏拦：'+value(summary['deadline']['miss_rate'])+'。',
           '- 合法对照误拦：'+value(summary['deadline']['false_positive_rate'])+'。',
           '这些是预设输出注入真实后处理函数的结果，不是真实大模型生成样本，不混入40题引用率。','',
           '## 已通过项','']
    lines += ['- '+key for key,passed in summary['acceptance_checks'].items() if passed] or ['- 无']
    lines += ['', '## 未通过门槛','']
    lines += ['- '+key for key,passed in summary['acceptance_checks'].items() if not passed] or ['- 无']
    lines += ['', '## 40题逐题结果','']
    for row in report['questions']:
        lines.append(f"- {row['id']}：{row['query']}；完成={row['complete']}，Doc={row['retrieval']['doc_hit']}，Article任一/全部={row['retrieval']['article_any']}/{row['retrieval']['article_all']}，预算后Article全部={row['assembly']['article_all']}，降级={row['degrade_correct']}；错误={row.get('error') or '无运行异常'}。")
    lines += ['', '## 真实失败案例摘录','']
    failures=[r for r in report['questions'] if not r['complete'] or r['retrieval']['doc_hit'] is False or r['retrieval']['article_all'] is False or r['degrade_correct'] is False or r.get('final_audit') and (r['final_audit']['false_citation'] or r['final_audit']['unverifiable'])]
    for row in failures[:8]:
        top='；'.join(r['title']+' / '+r['locator'] for r in row['top4'])
        excerpt=(row['raw_answer'] or '无原始输出')[:300].replace('\n',' / ')
        lines += [f"- {row['id']} Top4：{top or '无结果'}。原始输出摘录：{excerpt}"]
    if not failures:
        lines.append('- 未出现上述结构性失败；不代表法律结论已验收。')
    lines += ['', '## 防御失败明细','']
    lines += [f"- {r['id']}：{r['answer']}；预期拦截={r['expect_block']}，实际={r['actual_block']}。" for r in report['defense_tests'] if not r['correct']] or ['- 无']
    lines += ['', '## 证据与边界','',
              f"逐题JSON及网络事件在 `{report['run_directory']}`；每题含真实Qdrant响应payload、模型请求和原始响应流、实际材料、原始/最终答案及耗时。模型指纹和源码哈希见JSON报告。",
              '该评测仅覆盖固定40题及指定防御变体；引用率是可自动复现的显式编号/法律条号结构核验，不能解释为全部语义幻觉率或法律结论正确率。生产拦截器按专项安全修复要求升级；固定题目、预期、阈值和评分器保持不变。']
    if report.get('prior_incomplete_attempts'):
        lines += ['', '## 保留的不完整前次运行','']
        lines += [f"- {a['directory']}：已保存{a['status']['recorded_questions']}题；{a['status']['reason']}。" for a in report['prior_incomplete_attempts']]
    return '\n'.join(lines)+'\n'


async def run_evaluation(output,run_directory,timeout=180):
    """执行固定40题真实链路，每题落盘；初始化失败写失败证据并抛出，不补造分数。"""
    spec_path=ROOT/'data/eval/retrieval_eval_v2.json'
    spec,benchmark_hash=verify_frozen(spec_path,spec_path.with_suffix('.frozen.json'))
    run_directory.mkdir(parents=True,exist_ok=False)
    events=[]
    model=ObservedModel()
    await model.http.aclose()
    model.http=httpx.AsyncClient(base_url=model.base_url,trust_env=False,follow_redirects=False,
                                 timeout=httpx.Timeout(180,connect=5),transport=RecordingTransport(events,'ollama'))
    store=QdrantStore(transport=RecordingTransport(events,'qdrant'))
    started_at=datetime.now().astimezone().isoformat()
    results=[]
    try:
        retriever,coverage=await initialize(model,store,spec)
        tags=next(e['response_json'] for e in events if e['url'].endswith('/api/tags'))
        fingerprints={m['name']:m['digest'] for m in tags['models'] if m['name'] in {model.model,model.embedding}}
        paths=['local_rag/service.py','local_rag/retrieval.py','local_rag/ollama.py','local_rag/qdrant.py',
               'local_rag/ingest.py','prototype/rag/postprocess.py','prototype/rag/input_builder.py',
               'prototype/rag/citation_check.py','prototype/rag/deadline_check.py','prototype/rag/grounding.py',
               'local_rag/evaluation.py','local_rag/eval_runtime.py','local_rag/eval_judicial_evidence.py','scripts/run_rag_eval.py']
        metadata=dict(started_at=started_at,benchmark_sha256=benchmark_hash,corpus_id=store.corpus_id,coverage=coverage,
                      corpus_scope=retriever.evaluation_scope,
                      model_fingerprints=fingerprints,embedding_model_id=model.model_id,model=model.model,
                      generation_options=dict(temperature=0,num_ctx=8192,num_predict=650,think=False),question_timeout_seconds=timeout,
                      code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},bootstrap_network_events=list(events))
        metadata['prior_incomplete_attempts']=[dict(directory=str(p.parent),status=json.loads(p.read_text(encoding='utf-8'))) for p in sorted(run_directory.parent.glob('*/attempt_status.json'))]
        _save_json(run_directory/'metadata.json',metadata)
        defenses=run_defenses(spec,retriever.rows)
        _save_json(run_directory/'defense_results.json',dict(mode='deterministic_injection_no_model_call',results=defenses))
        observed=ObservedRetriever(retriever)
        service=ChatService(observed,model,timeout=timeout,session_dir=False)
        for question in spec['questions']:
            # 每题新会话，禁止上一题历史影响下一题，也不把预期条款传给检索器或模型。
            events.clear()
            model.last_messages=[]
            observed.hits=[]
            observed.mode=None
            observed.elapsed=0
            sid=service.create()
            start=time.perf_counter()
            rid=service.start(sid,question['query'])
            job=service.sessions[sid].jobs[rid]
            while not job['task'].done():
                await asyncio.wait({job['task']},timeout=20)
                print(question['id'],'phase',job['phase'],'elapsed',round(time.perf_counter()-start,1),flush=True)
            await job['task']
            status=service.result(sid,rid)
            result=status.get('result',{})
            raw,reasons=stream_answer(events)
            raw=result.get('raw_answer',raw)
            prompt=model.last_messages[-1]['content'] if model.last_messages else ''
            materials=result.get('retrieved',materials_in_prompt(observed.hits,prompt))
            query_calls=[e for e in events if e['service']=='qdrant' and e['url'].endswith('/points/query')]
            chat_calls=[e for e in events if e['service']=='ollama' and e['url'].endswith('/api/chat')]
            embed_calls=[e for e in events if e['service']=='ollama' and e['url'].endswith('/api/embed')]
            payload_recorded=bool(query_calls) and all(isinstance(e.get('response_json',{}).get('result',{}).get('points'),list) for e in query_calls)
            network_ok=bool(query_calls and chat_calls and embed_calls) and payload_recorded and all(e.get('status_code')==200 and e['complete'] for e in query_calls+chat_calls+embed_calls)
            complete=status['status']=='done' and observed.mode=='hybrid_qdrant' and network_ok and bool(reasons) and 'length' not in reasons
            has_raw=bool(chat_calls and raw)
            raw_audit=audit_citations(raw,materials,retriever.rows) if has_raw else None
            final_audit=audit_citations(result['answer'],materials,retriever.rows) if 'answer' in result else None
            row=dict(id=question['id'],query=question['query'],intent=question['intent'],domain=question['domain'],
                     expect_degrade=question['expect_degrade'],expected=question,complete=complete,job_status=status['status'],
                     error=status.get('error') or ('QDRANT_PAYLOAD_NOT_RECORDED' if not payload_recorded else None),elapsed_seconds=time.perf_counter()-start,retrieval_elapsed_seconds=observed.elapsed,
                     retrieval=score_hits(question,observed.hits),assembly=score_hits(question,materials),
                     raw_audit=raw_audit,final_audit=final_audit,
                     degrade_correct=judge_degrade(result.get('answer',''),result.get('blocked',False),question.get('scope_terms',[])) if question['expect_degrade'] and complete else False if question['expect_degrade'] else None,
                     retrieval_mode=observed.mode,top4=observed.hits,assembled_materials=materials,raw_answer=raw,
                     final_answer=result.get('answer'),blocked=result.get('blocked'),warnings=result.get('warnings',[]),
                     raw_output_blocked=result.get('raw_output_blocked',result.get('blocked')),
                     scope_degraded=result.get('scope_degraded',False),
                     model_stop_reasons=reasons,network_calls=dict(qdrant_query=len(query_calls),embedding=len(embed_calls),chat=len(chat_calls)),
                     network_events=list(events),model_messages=model.last_messages,token_usage=token_usage(events))
            _save_json(run_directory/(question['id']+'.json'),row)
            results.append(row)
            _save_json(run_directory/'progress.json',dict(completed_questions=len(results),last_id=question['id'],last_status=status['status'],updated_at=datetime.now().astimezone().isoformat()))
            service.sessions.pop(sid,None)
            print('SAVED',question['id'],'complete',complete,'doc',row['retrieval']['doc_hit'],'article',row['retrieval']['article_all'],'error',row['error'],flush=True)
        # 再次核对冻结字节及运行源码，禁止边跑边改评分器或生产代码。
        verify_frozen(spec_path,spec_path.with_suffix('.frozen.json'))
        for path,digest in metadata['code_sha256'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('运行期间源码变化，保留逐题证据但拒绝发布混合版本报告：'+path)
        report=dict(metadata,finished_at=datetime.now().astimezone().isoformat(),run_directory=str(run_directory),
                    summary=aggregate(results,defenses),questions=results,defense_tests=defenses)
        _save_json(run_directory/'report.json',report)
        _save_json(output,report)
        markdown=render_markdown(report)
        (run_directory/'report.md').write_text(markdown,encoding='utf-8')
        output.with_suffix('.md').write_text(markdown,encoding='utf-8')
        print('FINAL',json.dumps(report['summary'],ensure_ascii=False),flush=True)
        return report
    except BaseException as exc:
        _save_json(run_directory/'run_failure.json',dict(error=type(exc).__name__+': '+str(exc),recorded_questions=len(results),
                                                       network_events=list(events),failed_at=datetime.now().astimezone().isoformat()))
        raise
    finally:
        await model.close()
        await store.close()
