"""挑战基准独立只读校验器：字节冻结→封闭结构→用户事实定位→本地条文锚定。

入口 main；输入冻结JSON和 load_corpus 读取的已核验语料。输出控制台JSON、退出码0/1。
无网络、模型、Qdrant或文件写入，不读取题集自带参考正文作为真值。
此工具验证标注自洽，不能自动认证法理正确性或现行法律效力。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = ROOT/'data/benchmarks/benchmark_challenging_cases_v1.json'
FROZEN_SHA256 = '719970f07d10b1a2bc78fc293df02fd8594457d23a3a5e6a2a828e02da4cf248'
CORPUS_ID = '5b4d11d48704c5de789522d33a883f4747ed810edc049c5100ea875508839a26'
ANCHOR = {'law_name','article_number','version_id','chunk_id','text_sha256','source_url'}
CASE_KEYS = {'case_id','category','legal_domain','dialogue_history','query_message_index','query',
             'expected_core_articles','forbidden_trap_articles','extracted_facts','clarification_needed',
             'clarification_questions','rationale'}
TOP_KEYS = {'schema_version','dataset_id','corpus_id','corpus_rows_sha256','coverage','created_date',
            'provenance','official_review_urls','cases'}


def require(condition, message):
    """不用可被 python -O 禁用的assert实现验收门禁。"""
    if not condition:
        raise ValueError(message)


def fields(value, keys, label):
    """封闭字段集合，拒绝漏字段与未定义字段，防止隐藏真值未被核验。"""
    require(isinstance(value,dict) and set(value)==keys, label+'字段集合不符')


def text(value):
    return isinstance(value,str) and bool(value.strip())


def identity(article):
    """冲突检查跨版本按法律与条号保守判冲突，避免同号双重标签。"""
    return article['law_name'], article['article_number']


def validate_dataset(data, rows):
    """验证结构和实际语料；供CLI及所有破坏性测试共同调用，不改输入对象。

rows必须由调用方从可信本地资料加载，不能取自待检JSON；CLI固定使用load_corpus。
"""
    fields(data, TOP_KEYS, '题集')
    require(data['schema_version']=='1.0' and data['dataset_id']=='benchmark_challenging_cases_v1','版本错误')
    require(data['corpus_id']==CORPUS_ID,'批次错误')
    actual_hash = hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    require(data['corpus_rows_sha256']==actual_hash,'本地语料内容指纹变化')
    require(isinstance(data['coverage'],dict) and data['coverage'].get('chunks')==len(rows),'覆盖数量不符')
    require(text(data['provenance']) and data['created_date']=='2026-10-22','来源日期字段错误')
    require(isinstance(data['official_review_urls'],dict) and all(text(v) and v.startswith('https://') for v in data['official_review_urls'].values()),'审阅来源错误')
    cases = data['cases']
    require(isinstance(cases,list) and 15<=len(cases)<=18,'题数错误')
    categories = Counter(c.get('category') for c in cases if isinstance(c,dict))
    require(set(categories)=={'colloquial','negative_polarity','multi_turn'} and all(5<=v<=6 for v in categories.values()),'类别数量错误')
    by_id = {}
    for row in rows:
        require(row['id'] not in by_id,'语料存在重复切片ID')
        by_id[row['id']] = row
    case_ids = set()
    refs = 0
    for case in cases:
        fields(case, CASE_KEYS, '题目')
        cid = case['case_id']
        require(isinstance(cid,str) and re.fullmatch(r'CHALLENGE_\d{3}',cid) and cid not in case_ids,'题号重复或非法')
        case_ids.add(cid)
        require(all(text(case[k]) for k in ('legal_domain','query','rationale')),'题意说明为空')
        hist = case['dialogue_history']
        require(isinstance(hist,list),'历史必须为数组')
        require(len(hist) in (4,6) if case['category']=='multi_turn' else not hist,'历史轮数不符')
        for i,message in enumerate(hist):
            fields(message, {'message_index','role','content'},'历史消息')
            require(type(message['message_index']) is int and message['message_index']==i,'历史编号错误')
            require(message['role']==('user' if i%2==0 else 'assistant') and text(message['content']),'历史角色或文本错误')
        require(type(case['query_message_index']) is int and case['query_message_index']==len(hist),'当前问题编号错误')
        messages = hist+[dict(message_index=len(hist),role='user',content=case['query'])]
        require(type(case['clarification_needed']) is bool,'澄清标记必须为布尔')
        qs = case['clarification_questions']
        require(isinstance(qs,list) and all(text(q) for q in qs),'澄清问题格式错误')
        require(bool(qs)==case['clarification_needed'],'澄清标记和问题不一致')
        facts = case['extracted_facts']
        require(isinstance(facts,list) and len(facts)>0,'事实提取为空')
        fact_keys = set()
        for fact in facts:
            fields(fact, {'source_message_index','exact_text','start_char','end_char'},'事实')
            idx,start,end = (fact[k] for k in ('source_message_index','start_char','end_char'))
            require(all(type(v) is int for v in (idx,start,end)) and 0<=idx<len(messages),'事实消息编号或坐标错误')
            msg = messages[idx]
            require(msg['role']=='user','不能将助手输出作为用户事实')
            require(0<=start<end<=len(msg['content']) and text(fact['exact_text']),'事实范围错误')
            require(msg['content'][start:end]==fact['exact_text'],'事实不是原话连续字串')
            key=(idx,start,end)
            require(key not in fact_keys,'重复事实')
            fact_keys.add(key)
        if hist:
            require(any(f['source_message_index']<len(hist) for f in facts),'多轮题缺少历史用户事实锚点')
        positive, negative = [], []
        alternatives = Counter()
        for name in ('expected_core_articles','forbidden_trap_articles'):
            articles = case[name]
            require(isinstance(articles,list),'条文标注必须为数组')
            if name=='expected_core_articles': require(bool(articles),'核心条款为空')
            seen = set()
            for a in articles:
                core = name=='expected_core_articles'
                fields(a, ANCHOR|({'relation_type','alternative_group'} if core else {'level','trap_reason'}),'法条锚点')
                require(all(text(a[k]) for k in ANCHOR),'法条锚点字段为空')
                row = by_id.get(a['chunk_id'])
                require(row is not None,'切片ID不存在')
                require(a['law_name']==row['title'] and a['version_id']==row['version_id'] and a['source_url']==row['source_url'],'法律、版本或来源不匹配')
                # 原始定位第一段为法定条号；不把技术段落号转换为条号。
                require(a['article_number']==row['locator'].split()[0] and row['text'].startswith(a['article_number']),'法定条号不匹配')
                require(a['text_sha256']==hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),'正文哈希不匹配')
                require(identity(a) not in seen,'重复法条标注')
                seen.add(identity(a)); refs += 1
                if core:
                    relation = a['relation_type']
                    require(relation in {'mandatory','supplementary','alternative'},'关系类型非法')
                    if relation=='alternative':
                        require(text(a['alternative_group']),'备选组缺失')
                        alternatives[a['alternative_group']] += 1
                    else: require(a['alternative_group'] is None,'非备选条款不能指定备选组')
                    positive.append(identity(a))
                else:
                    require(a['level']=='forbidden_as_applicable_basis' and text(a['trap_reason']),'陷阱级别或理由错误')
                    negative.append(identity(a))
        require(all(n>=2 for n in alternatives.values()),'备选组至少需要两个不同条款')
        require(any(a['relation_type']=='mandatory' for a in case['expected_core_articles']) or bool(alternatives),'缺少必要依据或备选组')
        require(not set(positive)&set(negative),'期望条文与禁止适用条文冲突')
    return dict(status='PASS',cases=len(cases),categories=dict(categories),article_references=refs,
                corpus_rows=len(rows),scope='自洽性与本地锚定通过；非效力认证或检索成绩')


def verify_file(path=DATASET, rows=None):
    """校验固定字节哈希；不提供重写基准或跳过哈希的命令行选项。"""
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==FROZEN_SHA256,'冻结字节哈希不匹配，拒绝验收')
    if rows is None:
        from local_rag.retrieval import load_corpus
        rows, _ = load_corpus()
    return validate_dataset(json.loads(raw.decode('utf-8',errors='strict')), rows)


def main():
    """失败明确非零退出；仅打印报告，调用者可另存到新证据目录。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATASET)
    args=parser.parse_args()
    try:
        print(json.dumps(verify_file(args.dataset),ensure_ascii=False,indent=2))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='FAIL',error=type(exc).__name__+': '+str(exc)),ensure_ascii=False))
        return 1


if __name__=='__main__':
    sys.exit(main())
