"""回答法条校验入口：仅对本轮材料核对名称、条号及编号，不判断法律适用。

输入为原始回答和实际送入模型的材料；输出硬错误列表，无网络或文件副作用。
未标明法律名称的条号须结合同句或声明材料编号消歧，遇到虚构法名/条号或跨版本冲突时严格拦截。
"""
import re

ARTICLE = r'第[零〇一二三四五六七八九十百千万两\d]+条(?:之[零一二三四五六七八九十\d]+)?'


def numeric_value(value):
    """标准化中文或阿拉伯数值，支持两、〇、万，解析不依赖材料入库规则。"""
    if value.isdigit():
        return int(value)
    digits = dict(zip('零一二三四五六七八九', range(10)))
    digits.update({'〇': 0, '两': 2})
    total = section = current = 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif char == '万':
            total += (section + current) * 10000
            section = current = 0
        else:
            section += (current or 1) * {'十': 10, '百': 100, '千': 1000}[char]
            current = 0
    return total + section + current


def _clean_name(value):
    """只规范书名号/空白，不删除国别或版本等有意义信息。"""
    return re.sub(r'[\s《》]', '', value)


def law_aliases(materials):
    """名称归一化表，仅展开确定的基础法简称及本轮实际司法解释。"""
    aliases={}
    def add(alias,name):
        aliases.setdefault(_clean_name(alias),set()).add(name)
    for alias,name in {
        '民法典':'中华人民共和国民法典','劳动法':'中华人民共和国劳动法',
        '劳动合同法':'中华人民共和国劳动合同法','民诉法':'中华人民共和国民事诉讼法',
        '刑诉法':'中华人民共和国刑事诉讼法','民事诉讼法':'中华人民共和国民事诉讼法',
        '刑事诉讼法':'中华人民共和国刑事诉讼法',
    }.items():
        add(alias,name)
    for m in materials:
        name=m.get('title','')
        if not name:
            continue
        add(name,name)
        add(name.replace('中华人民共和国',''),name)
        if name.startswith('最高人民法院') and ('解释' in name or '规定' in name):
            for alias in ['解释','本解释','规定','本规定']:
                add(alias,name)
            if '民间借贷' in name:
                add('民间借贷规定',name)
            if '买卖合同' in name:
                add('买卖合同解释',name)
            if '合同编通则' in name:
                add('通则解释',name)
                add('合同编通则解释',name)
    return aliases


def resolve_law_names(prefix,materials):
    """返回紧邻名称的候选全称；None为无名称，空集合为显式未知名称。

    此处只解析名称，不判断条号通过；生产和独立审计各自核对实际材料及版本。
    """
    aliases=law_aliases(materials)
    book=re.search(r'《([^《》]+)》\s*$',prefix)
    if book:
        return aliases.get(_clean_name(book[1]),set())
    clean=_clean_name(prefix)
    # “民法典规定第X条”的规定是动词，不能覆盖前面的显式法名。
    if clean.endswith('规定'):
        prior=clean[:-2]
        direct=[a for a in aliases if a not in {'解释','本解释','规定','本规定'} and prior.endswith(a)]
        if direct:
            return aliases[max(direct,key=len)]
    direct=[a for a in aliases if clean.endswith(a)]
    if direct:
        alias=max(direct,key=len)
        # 泛称必须独立出现或接在明确的连接词后，不能吞掉虚构完整法名。
        if alias in {'解释','本解释','规定','本规定'}:
            before=clean[:-len(alias)]
            if before and not re.search(r'(?:和|及|与|依据|根据|依照|按照|适用|上述|该|[，,:：、（(])$',before):
                return set()
        return aliases[alias]
    books=re.findall(r'《([^《》]+)》',prefix)
    if books:
        return aliases.get(_clean_name(books[-1]),set())
    if re.search(r'(?:法典|法|条例|解释|规定|办法)$',clean):
        return set()
    return None


def article_grounding_issues(answer, materials):
    """名称/同句ID/条号/版本联合核对；裸条号允许同句/声明唯一材料或明确前文绑定。"""
    def key(value):
        suffix=re.sub(r'^.*条','',value)
        return numeric_value(re.sub(r'^第|条.*$','',value)),suffix
    issues=[]
    from prototype.rag.citation_check import USED_IDS_RE, parse_used_ids
    body = USED_IDS_RE.sub('', answer)
    normalized=re.sub(r'。\s*(\[M\s*\d+\])',r'\1。',body)
    provided_nums={m['material_id'][1:] for m in materials if m.get('material_id','').startswith('M')}
    declared_nums={mid[1:] for mid in (parse_used_ids(answer) or []) if mid[1:] in provided_nums}
    body_nums={n for n in re.findall(r'(?<![A-Za-z0-9])M\s*(\d+)', body)
               if n in provided_nums}
    fallback_nums=declared_nums or body_nums
    for clause in re.split(r'[\n。；]',normalized):
        refs=set(re.findall(r'(?<![A-Za-z0-9])M\s*(\d+)',clause))
        previous=None
        for match in re.finditer(ARTICLE,clause):
            adjacent=re.match(r'\s*((?:\[M\s*\d+\]\s*)+)',clause[match.end():])
            prefix=clause[:match.start()]
            named=resolve_law_names(prefix,materials)
            if adjacent:
                article_refs=set(re.findall(r'M\s*(\d+)',adjacent[1]))
            else:
                before_refs=re.findall(r'(?<![A-Za-z0-9])M\s*(\d+)',prefix)
                # 当条号前紧跟“根据M2第二十五条”且无显式法名时，优先绑定条号前最近的编号，避免被同句后半截的M1污染。
                article_refs={before_refs[-1]} if before_refs and named is None else refs
            if named is None:
                named=previous
            effective_refs=article_refs if article_refs else (fallback_nums if named is None else set())
            candidates=[m for m in materials if (named is None or m.get('title') in named)
                        and (not effective_refs or m['material_id'][1:] in effective_refs)]
            candidates=[m for m in candidates if any(key(match.group())==key(a) for a in re.findall(ARTICLE,m.get('locator','')))]
            signatures={(m.get('title'),m.get('version_id','')) for m in candidates}
            foreign=re.search(r'(德国|美国|英国|法国|日本|加拿大|澳大利亚|新加坡)',prefix)
            supported=len(signatures)==1 and (named is not None or bool(refs) or bool(fallback_nums)) and not foreign
            if not supported:
                issues.append(dict(type='UNSUPPORTED_ARTICLE',detail=f'法名、条号或材料绑定无法核对：{match.group()}，已拦截'))
            else:
                previous={next(iter(signatures))[0]}
        names=re.findall(r'《([^《》]+)》',clause)
        if names and not re.search(ARTICLE,clause) and re.search(r'规定|依据|根据',clause):
            aliases=law_aliases(materials)
            for name in names:
                options=aliases.get(_clean_name(name),set())
                versions={(m.get('title'),m.get('version_id','')) for m in materials if m.get('title') in options}
                if len(versions)!=1:
                    issues.append(dict(type='UNSUPPORTED_LAW',detail='回答依据的法律名称未包含在本轮材料中或指代不唯一'))
    return issues
