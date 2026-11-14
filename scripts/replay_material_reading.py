"""真实原始输出的许可重放：只读重建生产输入，逐字相同才重用已有4B原始回答。

主入口run；用于只改输出校验、不改模型提示/组装的回归。不调用4B，不冒充新推理。
只支持已完成的dev原始证据；使用生产ChatService及全部原检查，不改检索或数据库。
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.local-deps'),str(ROOT)]

from local_rag.daily_eval_runtime import execution_sources, load_dataset, make_scope, open_runtime, save_report, sha256_file
from local_rag.daily_evaluation import aggregate_answers, audit_answer, compare_answer_runs
from local_rag.eval_runtime import stream_answer
from local_rag.official_capture import fetch_official_npc_slice
from local_rag.service import ChatService
from scripts.eval_answers import ObservedRetriever


def verify_real_source(row):
    """核对源记录确有成功网络响应，响应全文等于raw；不把模拟或截断当作真实生成。"""
    events=row.get('network_chat') or []
    raw,reasons=stream_answer(events)
    if (row.get('split')!='dev' or not row.get('checks',{}).get('completed') or not events
            or not all(e.get('complete') and e.get('status_code')==200 for e in events)
            or not reasons or 'length' in reasons or raw!=row.get('raw_answer')):
        raise ValueError('源记录不是完整dev真实回答证据，拒绝重放')


class ReplayModel:
    """原模型请求正文完全相同才返回所存真实raw；没有HTTP连接、没有回答推理。"""
    def __init__(self, origin):
        self.origin=origin
        self.last_messages=None

    async def chat(self,messages):
        if messages!=self.origin['model_messages']:
            raise RuntimeError('重建输入与真实推理输入不同，不能重用原始回答')
        self.last_messages=[dict(m) for m in messages]
        return self.origin['raw_answer']


async def run(args):
    """读取真实dev报告、核对来源及同输入，重放完整服务；报告明确标记REPLAY，不含新网络证据。"""
    started=time.perf_counter()
    if not args.output_root.resolve().is_relative_to(ROOT):
        raise ValueError('报告只能写入项目内')
    origin=json.loads(args.source.read_text(encoding='utf-8'))
    for row in origin['results']:
        verify_real_source(row)
    source_hashes=execution_sources('scripts/replay_material_reading.py')
    async with open_runtime() as (retriever,_,info):
        dataset,digest=load_dataset(ROOT/'data/eval/daily_eval_v3.json',retriever.rows)
        if (info['degradation_reason'] or any(origin['scope'].get(k)!=v for k,v in
                (('dataset_sha256',digest),('corpus_id',info['corpus_id']),('embedding_id',info['embedding_id'])))
                or origin['runtime']['model_digest']!=info['model_digest']):
            raise RuntimeError('重放来源不一致')
        questions={q['id']:q for q in dataset['questions'] if q['split']=='dev'}
        observed=ObservedRetriever(retriever)
        results=[]
        for old in origin['results']:
            observed.mode='not_searched'
            q=questions.get(old['id'])
            if not q or q['query']!=old['query']:
                raise ValueError('题目与冻结dev不一致')
            model=ReplayModel(old)
            service=ChatService(observed,model,official_fetcher=fetch_official_npc_slice)
            sid=service.create();rid=service.start(sid,q['query'])
            await service.sessions[sid].jobs[rid]['task']
            job=service.result(sid,rid)
            if job['status']!='done' or model.last_messages is None:
                raise RuntimeError(job.get('error','未重建到模型入口'))
            result=job['result']
            if result['retrieved']!=old['materials']:
                raise RuntimeError('实际送入材料与源记录不一致')
            checks=audit_answer(q,result,result['retrieved'])
            results.append(dict(id=q['id'],split='dev',query=q['query'],category=q['category'],
                expect_scenario=q['expect_scenario'],expect_any=q['expect_any'],materials=result['retrieved'],
                raw_answer=old['raw_answer'],answer=result['answer'],result=result,checks=checks,
                mode=observed.mode,model_messages=model.last_messages,network_chat=[],
                generation='REPLAY_OF_REAL_RAW_NOT_NEW_INFERENCE',source_question_id=old['id']))
            print(f"已重放{len(results)}/{len(origin['results'])}；输入字节及完整材料相同",flush=True)
        if source_hashes!=execution_sources('scripts/replay_material_reading.py'):
            raise RuntimeError('重放运行中源码变化')
        report=dict(kind='answers',label='REPLAY_NOT_NEW_INFERENCE',
            scope=make_scope(info,digest,'',results),runtime=info,results=results,
            summary={'dev':aggregate_answers([r['checks'] for r in results])},
            source_report=str(args.source),source_report_sha256=sha256_file(args.source),
            source_hashes=source_hashes,input_identical_count=len(results),elapsed_seconds=time.perf_counter()-started,
            limitation='已有真实4B输出的离线许可复验；未产生新4B请求，不作推理耗时或生成质量增益结论。')
        report['answer_comparison']=compare_answer_runs(origin,report)
        directory=save_report(report,args.output_root)
        print(json.dumps(report['summary'],ensure_ascii=False,indent=2))
        print('REPLAY_REPORT=',directory)
        return 0 if report['answer_comparison']['comparable'] else 2


def main(argv=None):
    """命令入口；需显式给源报告，保留原始证据，只新建项目内重放报告。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,default=ROOT/'artifacts/local-rag/material-reading-experiment/replay')
    return asyncio.run(run(parser.parse_args(argv)))


if __name__=='__main__':
    raise SystemExit(main())
