"""端到端评测的独立评分口径：只读判分，不调用模型、不修改生产防御逻辑。

阅读顺序：verify_frozen → score_hits → audit_citations → aggregate。
输入为固化标注、真实检索材料及真实回答；输出可追溯命中/引用问题和分母。
引用审计是显式编号与法条引用的结构校验，不等同于法律结论或所有幻觉的语义判定。
"""
import hashlib
import json
import re

from .ingest import chinese_number

ARTICLE = re.compile(r'第([零〇一二三四五六七八九十百千万\d]+)条(之[零一二三四五六七八九十\d]+)?')
IDS = re.compile(r'(?<![A-Za-z0-9])M\s*(\d+)(?!\d)')


def verify_frozen(path, manifest):
    """核验评测原始字节哈希，改变题目或预期即拒绝执行；不写文件。"""
    raw = path.read_bytes()
    expected = json.loads(manifest.read_text(encoding='utf-8'))['sha256']
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('固化评测集哈希不匹配，拒绝运行或覆盖旧成绩')
    return json.loads(raw), expected


def article_key(text):
    """提取条号身份并保留增补条号；万级伪造编号仍作为原字符串比较，不抛异常。"""
    m = ARTICLE.search(text or '')
    if not m:
        return None
    def number(value):
        try:
            return int(value) if value.isdigit() else chinese_number(value)
        except (KeyError, ValueError):
            return value
    return number(m[1]), number(m[2][1:]) if m[2] else None


def score_hits(question, rows):
    """按文档名称及法律/版本/条号三元组评分；没有标注时返回None而非奖励成功。"""
    docs = set(question.get('expect_hit_docs', []))
    expected = {(r['title'],r['version_id'],article_key(r['article_no'])) for r in question.get('expect_hit_articles', [])}
    actual_docs = {r['title'] for r in rows}
    actual = {(r['title'],r['version_id'],article_key(r['locator'])) for r in rows}
    matched = expected & actual
    return dict(doc_hit=bool(docs & actual_docs) if docs else None,
                doc_all=docs <= actual_docs if docs else None,
                article_any=bool(matched) if expected else None,
                article_all=expected <= actual if expected else None,
                matched_articles=len(matched),expected_articles=len(expected))


def audit_citations(answer, materials, corpus):
    """审计原始或最终文本中显式引用，含USED_IDS行；不把程序生成卡片视为模型证据。

    不存在的编号/法律/条号或未提供的法条计作错误引用；无绑定的裸条号另记待核验。
    不识别隐含法律推理错误，不将本检测的0%描述成普遍无幻觉。
    """
    provided = {m['material_id']:m for m in materials}
    issues, unresolved = [], []
    for mid in sorted({f'M{m[1]}' for m in IDS.finditer(answer or '')}):
        if mid not in provided:
            issues.append(dict(type='UNKNOWN_MATERIAL_ID',value=mid))
    known_titles = {r['title'] for r in corpus}
    aliases = {t:t for t in known_titles}
    for title in known_titles:
        if title.startswith('中华人民共和国'):
            aliases.setdefault(title[len('中华人民共和国'):],title)
    # 当前语料中的全名优先，防止较短别名覆盖较长法规名。
    title_pattern = re.compile('|'.join(re.escape(t) for t in sorted(aliases,key=len,reverse=True))) if aliases else None
    body = re.sub(r'(?m)^\s*USED_IDS\s*[:：].*$', '', answer or '')
    statements=[]
    for part in re.findall(r'[^\n。；;]+[。；;]?',body):
        # 句号后单独写出的[M2]仍绑定前一句，不能因分句而漏掉错误引用关系。
        if statements and re.fullmatch(r'(?:\s*\[?\s*M\s*\d+\s*\]?\s*[,，、]?\s*)+',part):
            statements[-1]+=' '+part
        else:
            statements.append(part)
    for sentence in statements:
        if not sentence.strip():
            continue
        books = list(re.finditer(r'《([^》]{1,100})》',sentence))
        names = [(m.start(),aliases.get(m[1],m[1])) for m in books]
        if title_pattern:
            names.extend((m.start(),aliases[m[0]]) for m in title_pattern.finditer(sentence))
        refs = [provided[f'M{m[1]}'] for m in IDS.finditer(sentence) if f'M{m[1]}' in provided]
        for m in books:
            name = aliases.get(m[1],m[1])
            if name not in known_titles and (re.search(r'依据|根据|依照|按照|适用',sentence[:m.start()]) or re.search(r'规定|要求|明确|应当|可以',sentence[m.end():])):
                issues.append(dict(type='UNKNOWN_LAW_TITLE',value=m[1]))
        for match in ARTICLE.finditer(sentence):
            # 紧邻条号的编号优先绑定，避免同句其他编号替错误引用兜底。
            adjacent=re.match(r'\s*((?:\[M\s*\d+\]\s*)+)',sentence[match.end():])
            local_ids=set(IDS.findall(adjacent[1])) if adjacent else None
            article_refs=[r for r in refs if r['material_id'][1:] in local_ids] if local_ids is not None else refs
            if re.search(r'德国|美国|英国|法国|日本|加拿大|澳大利亚|新加坡',sentence[:match.start()]):
                issues.append(dict(type='FOREIGN_LAW_ARTICLE',value=match[0]))
                continue
            preceding = [p for p in names if p[0] < match.start()]
            title = max(preceding,key=lambda p:p[0])[1] if preceding else None
            key = article_key(match[0])
            # 共享名称词典而不共享放行结论；独立核对语料、材料、条号和版本。
            from prototype.rag.grounding import resolve_law_names
            resolved=resolve_law_names(sentence[:match.start()],materials)
            if resolved is not None:
                if not resolved:
                    issues.append(dict(type='UNKNOWN_LAW_TITLE',value=match[0]))
                    continue
                choices=[r for r in materials if r['title'] in resolved and article_key(r['locator'])==key
                         and (not article_refs or r in article_refs)]
                signatures={(r['title'],r.get('version_id','')) for r in choices}
                if len(signatures)>1:
                    issues.append(dict(type='AMBIGUOUS_ARTICLE_REFERENCE',value=match[0]))
                    continue
                title=next(iter(signatures))[0] if signatures else next(iter(resolved)) if len(resolved)==1 else None
                if title is None:
                    issues.append(dict(type='ARTICLE_NOT_IN_PROVIDED_MATERIAL',value=match[0]))
                    continue
            if title:
                exists = any(r['title']==title and article_key(r['locator'])==key for r in corpus)
                supported = any(r['title']==title and article_key(r['locator'])==key for r in materials)
                if not exists or not supported:
                    issues.append(dict(type='NONEXISTENT_ARTICLE' if not exists else 'ARTICLE_NOT_IN_PROVIDED_MATERIAL',title=title,value=match[0]))
                elif article_refs and not any(r['title']==title and article_key(r['locator'])==key for r in article_refs):
                    issues.append(dict(type='NAMED_ARTICLE_WRONG_MATERIAL_ID',title=title,value=match[0]))
            elif article_refs:
                identities={(r['title'],r.get('version_id','')) for r in article_refs if article_key(r['locator'])==key}
                if len(identities)>1:
                    issues.append(dict(type='AMBIGUOUS_ARTICLE_REFERENCE',value=match[0]))
                elif not identities:
                    issues.append(dict(type='ARTICLE_MATERIAL_MISMATCH',value=match[0]))
            else:
                unresolved.append(dict(type='UNBOUND_ARTICLE',value=match[0]))
    return dict(false_citation=bool(issues),unverifiable=bool(unresolved),issues=issues,unverifiable_issues=unresolved,
                scope='显式材料编号/法律条号结构核验；未判断全部法律语义及隐含幻觉')


def judge_degrade(answer, blocked, scope_terms):
    """保守判定范围降级：须明确指出指定地域/规则资料缺口；通用拦截提示不算成功。"""
    if blocked or not any(term in answer for term in scope_terms):
        return False
    gap = re.search(r'未收录|未覆盖|未提供|缺少|缺乏|无(?:对应|相关|相应)?(?:依据|资料|材料|规则)|(?:细则|规则|材料).{0,12}未在.{0,12}(?:体现|包含)|没有.{0,15}(?:依据|资料|材料|规则)|无法.{0,12}(?:确认|核验|核实|判断)|需.{0,12}(?:查询|查阅|核实)',answer)
    refusal = re.search(r'(?:无法|不能|尚不能|不足以).{0,12}(?:确认|确定|给出|提供|核实|核验|判断|回答)',answer)
    definite = re.search(r'(?:必须|应当|有权|可以申请|补贴为).{0,16}(?:\d+|[一二三四五六七八九十百]+)\s*(?:日|天|月|年|元)',answer)
    concrete = re.search(r'(?:期限|通知期|补贴|收入门槛|申请条件|金额).{0,12}(?:为|是|要求|需|至少|不得超过).{0,14}(?:\d+\s*(?:日|天|月|年|元)|本区户籍|本地户籍|社保缴纳|收入低于)',answer)
    return bool(gap and refusal) and not (definite or concrete)


def ratio(numerator, denominator):
    """保留原始分子分母；没有实际样本时rate为None，禁止记0%或100%通过。"""
    return dict(numerator=numerator,denominator=denominator,rate=numerator/denominator if denominator else None)


def aggregate(results, defenses, expected_total=40):
    """分层汇总并执行预先声明门槛；异常保留分母，不当作安全降级或空引用成功。"""
    def layer(name):
        scored=[r[name] for r in results if r[name]['doc_hit'] is not None]
        articles=[r[name] for r in results if r[name]['article_any'] is not None]
        return dict(doc_hit_at4=ratio(sum(s['doc_hit'] for s in scored),len(scored)),
                    doc_all_at4=ratio(sum(s['doc_all'] for s in scored),len(scored)),
                    article_any_at4=ratio(sum(s['article_any'] for s in articles),len(articles)),
                    article_all_at4=ratio(sum(s['article_all'] for s in articles),len(articles)))
    raw=[r['raw_audit'] for r in results if r.get('raw_audit') is not None]
    final=[r['final_audit'] for r in results if r.get('final_audit') is not None]
    boundary=[r for r in results if r['expect_degrade']]
    negative=[r for r in defenses if r['kind']=='deadline' and r['expect_block']]
    positive=[r for r in defenses if r['kind']=='deadline' and not r['expect_block']]
    completed=sum(bool(r['complete']) for r in results)
    retrieval,assembly=layer('retrieval'),layer('assembly')
    boundary_score=ratio(sum(bool(r['degrade_correct']) for r in boundary),len(boundary))
    intercepted=ratio(sum(bool(r['actual_block']) for r in negative),len(negative))
    missed=ratio(sum(not r['actual_block'] for r in negative),len(negative))
    false_positive=ratio(sum(bool(r['actual_block']) for r in positive),len(positive))
    checks=dict(real_execution_complete=completed==expected_total and len(results)==expected_total,
                doc_hit_at4=(retrieval['doc_hit_at4']['rate'] or 0)>=.85,
                final_false_citation_zero=len(final)==expected_total and not any(a['false_citation'] or a['unverifiable'] for a in final),
                boundary_all_correct=bool(boundary) and boundary_score['rate']==1,
                deadline_interception_100=bool(negative) and intercepted['rate']==1,
                deadline_false_positive_zero=bool(positive) and false_positive['rate']==0)
    verdict='合格' if all(checks.values()) else '部分通过' if checks['real_execution_complete'] and any(checks.values()) else '不达标'
    return dict(execution=dict(expected=expected_total,recorded=len(results),completed=completed,failed=len(results)-completed),
                retrieval=retrieval,assembly=assembly,
                citations=dict(raw_false_rate=ratio(sum(a['false_citation'] for a in raw),len(raw)),
                               final_false_rate=ratio(sum(a['false_citation'] for a in final),len(final)),
                               raw_unverifiable=sum(a['unverifiable'] for a in raw),final_unverifiable=sum(a['unverifiable'] for a in final)),
                boundary=boundary_score,deadline=dict(interception=intercepted,miss_rate=missed,false_positive_rate=false_positive),
                acceptance_checks=checks,verdict=verdict)
