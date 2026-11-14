"""本地知识库检索：加载资料、维护索引，以两路粗召回及救济结构/专门范围精排生成候选。
load_corpus()读取资料，build()向量入库，search()查找候选，tail_candidates()取页面"更多候选"；检索命中不等于法律结论成立。
match_official_whitelist_catalog()仅定位缺失的官方目录标题，不抓正文、不写入检索库。

阅读指南：docs/guides/PYCHARM_PROJECT_READING.md（相对项目根目录）。
"""
import asyncio
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

from .official_capture import catalog_entry_metadata, normalize_catalog_title

ROOT = Path(__file__).resolve().parents[1]
# 每路粗召回最多20条，让第13—20名也能参与结构精排；最终返回数量仍由 Top-K 控制。
CANDIDATE_LIMIT = 20
# 页面"更多候选"看两路各自全库前60名；只供展示，不进入 search() 的融合，也不送模型。
TAIL_DEPTH = 60
# 文档向量化格式：只决定“送给 embedding 的文本”，不改 rows 里的 text，也不改展示、送模和校验原文。
# plain 是生产默认（逐字节沿用旧缓存与旧集合）；law_header_v1 在正文前加“《法名》条号”。
DOC_FORMAT_PLAIN = "plain"
DOC_FORMAT_LAW_HEADER_V1 = "law_header_v1"
DOC_FORMATS = (DOC_FORMAT_PLAIN, DOC_FORMAT_LAW_HEADER_V1)


def document_embedding_text(row, doc_format=DOC_FORMAT_PLAIN):
    """返回单条资料送去向量化的文本；只读字段，不修改 row 本身。

    plain: 只送正文，保持既有缓存、集合与生产成绩不变。
    law_header_v1: 送 "《title》locator\\ntext"，让法名和条号参与语义匹配。
    注意 locator 当前含“正文提取行L###”来源标记，本实验按字面格式一并送入，结果中如实记录。
    """
    text = row.get("text", "")
    if doc_format == DOC_FORMAT_PLAIN:
        return text
    if doc_format == DOC_FORMAT_LAW_HEADER_V1:
        return f"《{row.get('title', '')}》{row.get('locator', '')}\n{text}"
    raise ValueError(f"未知文档向量化格式：{doc_format}")


# 权利保护/直接行为制裁也是实体规范结构；仅识别句法，不写死法名，不推断个案成立。
PROTECTIVE_STRUCT_RE = re.compile(r"享有[^。；\n]{0,8}权[。；：]|任何组织(?:或者|和)个人不得")
# 量刑句只证明处罚后果；须有词面支持或处于语义前两名，才作为直接制裁结构额外提权。
DIRECT_SANCTION_RE = re.compile(r"处[^。；\n]{0,24}有期徒刑")
# 通用责任兜底只说明违反该规范的后果，没有给出具体行为；缺少词面支持时不据此额外提权。
GENERIC_LIABILITY_RE = re.compile(r"违反本(?:法|条例|规定|办法)(?:规定)?[，,][^。；\n]*承担民事责任")
OFFICIAL_CATALOG_PATH = ROOT / 'data/catalogs/official-20261020/npc/judicial/catalog.json'
# 目录标题的程序性套话不作为专题证据；保留建设工程、财产保全、证据等真正主题词。
CATALOG_TITLE_BOILERPLATE = (
    '最高人民法院', '最高人民检察院', '中华人民共和国', '人民法院', '具体应用法律', '具体适用法律', '适用法律问题', '适用法律',
    '若干问题', '若干', '关于', '审理', '办理', '案件', '纠纷', '规定', '解释', '通知', '意见', '办法', '有关', '的',
)
CATALOG_GENERIC_PHRASES = {'民事诉讼', '行政诉讼', '刑事诉讼', '合同纠纷', '法律问题', '法律规定'}
# 泛立案口语虽补两类专题，目录联网还须原文有对应行为线索，不能把其他罪名硬套为诈骗/盗窃。
CATALOG_CRIMINAL_CONTEXT = (
    ('诈骗刑事', ('诈骗', '被骗', '骗钱', '骗了', '骗走')),
    ('盗窃刑事', ('盗窃', '偷钱', '偷东西', '偷走')),
)


@lru_cache(maxsize=1)
def load_official_whitelist_catalog():
    """只读并缓存已保存的默认官方目录；缺失/损坏时返回空，不联网、不写法规或索引。

    更新目录快照后需重启服务或显式清除本函数缓存；目录存在不代表全文已入库。
    """
    try:
        value = json.loads(OFFICIAL_CATALOG_PATH.read_text(encoding='utf-8'))
        entries = value.get('entries', []) if isinstance(value, dict) else value
        return tuple(entry for entry in entries if isinstance(entry, dict)) if isinstance(entries, list) else ()
    except (OSError, ValueError):
        return ()


def match_official_whitelist_catalog(question, catalog_entries, indexed_titles, top_k=2):
    """纯函数匹配未入库的官方现行目录标题，返回显著相关的安全元数据；不抓正文、不写盘。

    原问题或明确口语扩展至少连续四字命中非套话主题；扩展命中也须有足够双字重合与覆盖。
    标题相关不证明条文适用；相似候选只保留接近最佳匹配者，同名不重复推荐。
    """
    if top_k <= 0:
        return []
    indexed = {normalize_catalog_title(title) for title in indexed_titles}
    question_key = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', normalize_catalog_title(question))
    expanded = expand_colloquial_query(question)
    expanded_key = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', normalize_catalog_title(expanded))
    original_terms, expanded_terms = set(tokens(question)), set(tokens(expanded))
    ranked = []
    for position, entry in enumerate(catalog_entries):
        metadata = catalog_entry_metadata(entry)
        if metadata is None or normalize_catalog_title(metadata['title']) in indexed:
            continue
        core = normalize_catalog_title(metadata['title'])
        for phrase in CATALOG_TITLE_BOILERPLATE:
            core = core.replace(phrase, '')
        core = re.sub(r'\([一二三四五六七八九十\d]+\)$', '', core)
        core = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', core)
        if any(topic in core and not any(hint in question_key for hint in hints)
               for topic, hints in CATALOG_CRIMINAL_CONTEXT):
            continue
        fragments = [core[i:i+4] for i in range(len(core) - 3) if core[i:i+4] not in CATALOG_GENERIC_PHRASES]
        original_hit = any(fragment in question_key for fragment in fragments)
        expanded_hit = any(fragment in expanded_key for fragment in fragments)
        if not original_hit and not expanded_hit:
            continue
        title_terms = set(tokens(core))
        # 原问句部分命中不能阻断完整扩展主题；覆盖用两者并集，原文词项另加权以区分诈骗/盗窃等同模板标题。
        query_terms = original_terms | expanded_terms
        overlap = len(query_terms & title_terms)
        if overlap < 3 or overlap / max(len(title_terms), 1) < 0.25:
            continue
        score = (overlap + 0.5 * len(original_terms & title_terms)
                 + 0.1 * len(expanded_terms & title_terms)) / math.sqrt(len(title_terms))
        ranked.append((score, metadata['publication_date'], metadata['effective_date'], position, metadata))
    ranked.sort(key=lambda row: (-row[0], -int(row[1].replace('-', '') or '0'),
                                 -int(row[2].replace('-', '') or '0'), row[3]))
    matches, seen = [], set()
    best = ranked[0][0] if ranked else 0
    for score, _, _, _, metadata in ranked:
        key = normalize_catalog_title(metadata['title'])
        if key in seen or score < best * 0.85:
            continue
        matches.append(dict(metadata, match_score=round(score, 4)))
        seen.add(key)
        if len(matches) >= top_k:
            break
    return matches

# 权利救济的规范结构，仅调节混合检索的向量贡献，不证明条款与案情实际相符。
REMEDY_STRUCT_MARKERS = (
    "享有下列权利", "有权请求", "提出申诉", "依法提起诉讼", "承担民事责任", "赔偿损失", "停止侵害",
    "有权查阅", "有权复印", "不得采取停止供电", "准予毕业", "应当予以返还", "请求人民法院", "申请复核",
    "履行更换、修理", "应当受理",
)
# 普通维权问题中容易错位的内部监管/证书查处结构；同时含救济词时仍先按本组处理。
MISMATCHED_OFFICIAL_MARKERS = (
    "宣布证书无效", "弄虚作假或者剽窃他人教学成果", "有关部门的工作人员违反本法规定，滥用职权",
    "在教师资格认定工作中玩忽职守", "监督管理职责的部门及其工作人员，必须对在履行职责中知悉",
    "有关主管部门及其工作人员对在履行职责中知悉",
)
# 原问句明确涉及机关履职或假证时关闭结构调节；只是关键词开关，不识别否定或复杂主体关系。
OFFICIAL_QUERY_KEYWORDS = (
    "部门", "机关", "工作人员", "履职", "履行职责", "执法", "调查", "政务", "伪造", "假证", "弄虚作假",
)

# 专门主体/工程法规缺少原问句适用线索时降低混合权重；只是关键词门控，不认定适用与否。
# 仅用于混合精排，不删除候选、不影响纯词面降级；英文线索按小写比较。
SPECIAL_SCOPE_RULES = (
    (("未成年人保护法", "预防未成年人犯罪法"), ("未成年", "儿童", "小孩", "孩子", "幼儿", "小学生", "中学生", "父母", "监护人")),
    (("工会法",), ("工会",)),
    (("娱乐场所管理条例",), ("娱乐场所", "歌舞", "游艺", "ktv", "酒吧", "网吧")),
    (("归侨侨眷",), ("归侨", "侨眷", "华侨")),
    (("三峡工程", "水利水电工程", "蓄滞洪区"), ("三峡", "水利", "水电", "水库", "蓄滞洪", "移民")),
    (("出国旅游",), ("出国", "出境", "境外", "国外", "领队")),
    (("国有企业富余职工",), ("国有企业", "国企", "富余职工")),
)

# 口语提问到法定规范用语的确定性同义词映射表：
# 仅在用户问题包含左侧口语词时，追加右侧法律概念词辅助检索，不篡改用户原话事实。
# 扩展词只供 BM25 召回候选，不代表构成要件成立；向量检索仍使用用户原问句。
COLLOQUIAL_ALIASES = {
    "不给钱": "拖欠 支付 报酬 债务",
    # 劳动合同法第三十八条：同时补充违法事由和劳动者解除权，避免只召回补偿条款。
    "不发工资": "拖欠克扣工资 未及时足额支付劳动报酬 解除劳动合同",
    **dict.fromkeys(("拖欠工资", "不给工资", "不结工资", "不给加班费", "不结加班", "拖欠加班费"),
                    "未及时足额支付劳动报酬 解除劳动合同 经济补偿"),
    "克扣工资": "未及时足额支付劳动报酬 解除劳动合同",
    # 同场景共用规范表述；短触发词覆盖“还不给我交社保”等夹字说法，不判断事实真伪。
    **dict.fromkeys(("不交社保", "没交社保", "不缴社保", "没缴社保", "不买社保", "没买社保", "交社保", "缴社保"),
                    "未依法为劳动者缴纳社会保险费 解除劳动合同"),
    **dict.fromkeys(("经济补偿", "按工作年限"),
                    "经济补偿按劳动者在本单位工作的年限 每满一年支付一个月工资的标准向劳动者支付 "
                    "劳动者月工资高于用人单位所在直辖市"),
    "没签合同": "未订立书面劳动合同 二倍工资",
    "不签合同": "未订立书面劳动合同 二倍工资",
    "辞退": "解除劳动合同 违法解除 赔偿金 经济补偿",
    "开除": "解除劳动合同 严重违反规章制度",
    "裁员": "经济性裁员 解除劳动合同 经济补偿",
    # 工伤停工留薪：只认“干活摔/干活受伤/工伤休息”等工作场景动作，不认裸词“骨折”；
    # 是否属于工伤、停工留薪期多长仍须核对事故经过、诊断和认定材料。
    **dict.fromkeys(("干活摔", "干活受伤", "工伤休息"),
                    "停工留薪期 原工资福利待遇不变 由所在单位按月支付 需要暂停工作接受工伤医疗"),
    "借钱": "民间借贷 借款合同 返还借款",
    "欠钱": "借款 逾期利息 返还",
    "不还钱": "逾期还款 返还借款 逾期利息",
    # 合同违约按具体履行问题补词，不将一般违约直接推定为合同无效或可解除。
    "不按合同办": "不履行合同义务 违约责任 继续履行 赔偿损失",
    "货不对板": "质量不符合约定 修理 更换 退货 违约责任",
    "交了钱不发货": "迟延履行 交付标的物 违约责任",
    # 催告解约、解除后返还与替换材料分开补词，是否满足解除/补救条件仍须核对事实。
    **dict.fromkeys(("一直不送", "催好多次", "迟迟不发"),
                    "当事人一方迟延履行主要债务，经催告后在合理期限内仍未履行 当事人可以解除合同"),
    **dict.fromkeys(("取消装修", "取消合同", "半途取消"),
                    "合同解除后，尚未履行的，终止履行 "
                    "已经履行的，根据履行情况和合同性质，当事人可以请求恢复原状或者采取其他补救措施，并有权请求赔偿损失"),
    **dict.fromkeys(("换了牌子", "偷换材料"),
                    "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任"),
    **dict.fromkeys(("配合验收", "拒绝配合", "协助义务"),
                    "按照约定全面履行自己的义务 遵循诚信原则 根据合同的性质 "
                    "目的和交易习惯履行通知 协助 保密等义务"),
    **dict.fromkeys(("不履行合同", "调低违约金", "违约金比实际损失高"),
                    "当事人一方不履行合同义务或者履行合同义务不符合约定 "
                    "承担继续履行 采取补救措施或者赔偿损失等违约责任 "
                    "约定的违约金过分高于造成的损失 请求人民法院或者仲裁机构予以适当减少"),
    **dict.fromkeys(("约定还款日", "一直不还", "到期不还"),
                    "借款人应当按照约定的期限返还借款 对借款期限没有约定或者约定不明确 "
                    "借款人未按照约定的期限返还借款 应当按照约定或者国家有关规定支付逾期利息"),
    **dict.fromkeys(("还没有法院判决", "已经是罪犯", "未经法院判决"),
                    "未经人民法院依法判决 对任何人都不得确定有罪"),
    # 不用多义裸词“顶嘴”判断行政争议；仍只认带申辩结果的“罚更多/再顶嘴”，
    # 并只补明确的陈述、申辩与不得加重处罚概念。
    **dict.fromkeys(("不服申辩", "越解释越罚", "罚更多", "再顶嘴"),
                    "当事人有权进行陈述和申辩 行政机关必须充分听取当事人的意见 "
                    "行政机关不得因当事人陈述、申辩而给予更重的处罚"),
    **dict.fromkeys(("不给答复", "办证窗口", "拖着不批", "拆迁补偿", "赔迁", "征地补偿"),
                    "申请行政许可，行政机关拒绝或者在法定期限内不予答复 "
                    "对征收、征用决定及其补偿决定不服 提起行政诉讼"),
    # 行政处罚后的起诉期限：只认收到/接到处罚、不服处罚、起诉期限等带救济动作的短语，
    # 不认裸词“处罚/法院”；是否超过六个月仍须核对知道或应当知道行政行为的日期。
    **dict.fromkeys(("接到处罚", "收到处罚", "不服处罚", "起诉期限"),
                    "应当自知道或者应当知道作出行政行为之日起六个月内提出 "
                    "直接向人民法院提起诉讼 起诉期限"),
    "撞伤": "机动车交通事故责任 人身损害赔偿",
    "车祸": "机动车交通事故责任 医疗费 误工费",
    "被打伤": "侵害他人人身权益 人身损害赔偿 医疗费 误工减少的收入",
    "被狗咬": "饲养的动物造成他人损害 动物饲养人 管理人 侵权责任",
    "网上造谣": "诽谤 名誉权 停止侵害 消除影响 恢复名誉",
    **dict.fromkeys(("闯进来", "私自进屋", "乱进我屋"),
                    "除法律另有规定或者权利人明确同意外，任何组织或者个人不得实施下列行为 "
                    "进入、拍摄、窥视他人的住宅、宾馆房间等私密空间"),
    # 婚姻继承词仅作召回线索，不能据此认定所有恋爱消费均为彩礼或可返还赠与。
    "闹离婚": "感情确已破裂 调解无效 准予离婚",
    "家暴": "家庭暴力 人身安全保护令 实施家庭暴力 申请人身安全保护令 离婚损害赔偿",
    "离婚分财产": "夫妻共同财产 协议不成 人民法院判决",
    "不给抚养费": "不直接抚养子女 抚养费 子女抚养",
    "老人去世房子归谁": "遗产 法定继承 继承顺序 配偶 子女 父母",
    "没留遗嘱": "法定继承 继承顺序 遗产",
    **dict.fromkeys(("嫁出去", "外嫁女", "重男轻女"),
                    "遗产按照下列顺序继承 第一顺序：配偶、子女、父母 继承权男女平等"),
    # 婚前房产：只认“结婚前/婚前买房/婚前财产”等财产归属表述，不认裸词“房子”；
    # 是否属于个人财产仍须核对取得时间、出资和登记事实。
    **dict.fromkeys(("结婚前", "婚前买房", "婚前财产"),
                    "夫妻一方的个人财产 一方的婚前财产 不因婚姻关系的延续而转化为夫妻共同财产"),
    # 数份遗嘱冲突：只认“两张/两份/多份遗嘱、遗嘱不一样/冲突”等并存冲突表述，
    # 不认裸词“遗嘱”；哪份为准仍须核对形式、时间和内容抵触事实。
    **dict.fromkeys(("两张遗嘱", "两份遗嘱", "多份遗嘱", "遗嘱不一样", "遗嘱冲突", "前后遗嘱", "几份遗嘱"),
                    "立有数份遗嘱 内容相抵触的 以最后的遗嘱为准 遗嘱人可以撤回、变更自己所立的遗嘱"),
    "退货": "七日内无理由退货 消费者权益保护",
    "假货": "欺诈 惩罚性赔偿 退一赔三",
    "定金": "定金罚则 双倍返还",
    # 补足刑事分档和折价返还的原文词项，避免共同的意思表示用语挤掉刑事或返还候选。
    # 分档词只用于匹配，不据用户口语金额判断刑事门槛，也不预先认定全部支出可返还。
    **dict.fromkeys(("骗了", "被骗"),
                    "诈骗公私财物 数额较大的 处三年以下有期徒刑 "
                    "一方以欺诈手段 使对方在违背真实意思的情况下实施的民事法律行为 "
                    "受欺诈方有权请求人民法院或者仲裁机构予以撤销 "
                    "行为人因该行为取得的财产 应当予以返还 "
                    "拘役或者管制 并处或者单处罚金 数额巨大或者有其他严重情节 "
                    "数额特别巨大或者有其他特别严重情节 "
                    "不能返还或者没有必要返还的 应当折价补偿"),
    # 保留诈骗与民事返还两路；款项是否为借款、附义务赠与或彩礼，仍须客观事实区分。
    **dict.fromkeys(("对象", "男朋友", "女朋友", "前任", "花了我", "彩礼", "分手"),
                    "当事人请求返还按照习俗给付的彩礼 双方未办理结婚登记手续 "
                    "赠与可以附义务 赠与附义务的 受赠人应当按照约定履行义务 "
                    "民事法律行为无效 被撤销或者确定不发生效力后 "
                    "行为人因该行为取得的财产 应当予以返还"),
    # 年龄/称谓只补简短的行为能力概念，不把身份线索当成消费行为，也不据称谓认定具体年龄。
    **dict.fromkeys(("十岁", "八岁", "九岁", "未成年", "小孩", "熊孩子"),
                    "八周岁 限制民事行为能力 法定代理人同意、追认"),
    # 有打赏、游戏充值或直播场景才补付费服务与返还规则；仍须核对年龄、行为与同意事实。
    **dict.fromkeys(("刷礼物", "直播打赏", "给主播刷", "游戏充值", "游戏充钱", "直播"),
                    "八周岁以上的未成年人为限制民事行为能力人 "
                    "实施民事法律行为由其法定代理人代理或者经其法定代理人同意、追认 "
                    "限制民事行为能力人实施的纯获利益的民事法律行为或者与其年龄、智力、精神健康状况相适应的民事法律行为有效 "
                    "民事法律行为无效、被撤销或者确定不发生效力后，行为人因该行为取得的财产，应当予以返还 网络游戏、网络直播 "
                    "不得向未成年人提供与其民事行为能力不符的付费服务"),
    # 房屋隐瞒争议同时召回欺诈、缔约过失与中介报告义务，不据死亡事实当然认定可退房。
    **dict.fromkeys(("死过人", "凶宅", "隐瞒", "中介没说", "房东没说"),
                    "一方以欺诈手段，使对方在违背真实意思的情况下实施的民事法律行为，受欺诈方有权请求人民法院或者仲裁机构予以撤销 "
                    "故意隐瞒与订立合同有关的重要事实或者提供虚假情况 中介人应当就有关订立合同的事项向委托人如实报告 "
                    "中介人故意隐瞒与订立合同有关的重要事实或者提供虚假情况，损害委托人利益的，不得请求支付报酬并应当承担赔偿责任"),
    **dict.fromkeys(("半路加价", "坐地起价", "加钱才继续干", "不给就停工", "全包合同",
                     "装修全包", "不加就停工", "中途加价", "临时加价"),
                    "当事人协商一致，可以变更合同 "
                    "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任"),
    # 旅游消费线索不自动触发出境专门法门控；是否满足退货条件仍须核对实际材料与行程。
    **dict.fromkeys(("强制购物", "逼我们买", "低价旅游", "低价团", "赶下大巴", "甩客"),
                    "旅行社组织、接待旅游者，不得指定具体购物场所，不得安排另行付费旅游项目 "
                    "导游人员进行导游活动，不得欺骗、胁迫旅游者消费或者与经营者串通欺骗、胁迫旅游者消费 "
                    "旅游者有权在旅游行程结束后三十日内，要求旅行社为其办理退货并先行垫付退货货款"),
    # 被指控与诬告是不同事实判断；同时召回有罪/无罪取证规则，不能将用户否认直接当作诬告成立。
    # 保留既有公开捏造事实的治安分支；只保留诬告词串会使相关条文错失多短语提权，不预判指控去向。
    **dict.fromkeys(("告我强奸", "告我猥亵", "诬告", "陷害", "栽赃", "冤枉我", "反咬一口", "仙人跳",
                     "冤死", "假报案", "编造报案"),
                    "捏造事实诬告陷害他人，意图使他人受刑事追究，情节严重的 "
                    "不是有意诬陷，而是错告，或者检举失实的 "
                    "捏造事实诬告陷害他人，企图使他人受到刑事追究或者受到治安管理处罚的 "
                    "收集能够证实犯罪嫌疑人、被告人有罪或者无罪、犯罪情节轻重的各种证据 "
                    "公然侮辱他人或者捏造事实诽谤他人"),
    # 初次被叫去问话：只认“叫我去问话/喊我去问话”等侦查到场表述，
    # 不认裸词“律师/问话”；是否已立案、是否在押仍须核对事实。
    **dict.fromkeys(("叫我去问话", "喊我去问话"),
                    "第一次讯问 有权委托辩护人 侦查期间 只能委托律师作为辩护人"),
    # 隐晦表述补充不限定一般受害人性别的强制猥亵线索，不据口语认定具体行为或罪名。
    # 不追加暴力/胁迫等通用前缀，避免无谓抬高其他罪名的词面分数。
    **dict.fromkeys(("把我给那个了", "强迫我发生", "被强暴", "被性侵", "强制猥亵", "猥亵我",
                     "乱摸", "非礼", "摸我"),
                    "强制猥亵他人或者侮辱妇女的 "
                    "聚众或者在公共场所当众犯前款罪的 猥亵他人的"),
    # 复合技术场景补充行为与要件词，不据口语断定联网违法、软件权属或商业秘密成立。
    "vpn": "计算机信息网络直接进行国际联网 国际出入口信道 自行建立或者使用其他信道",
    "窃取我的项目": "侵犯商业秘密 盗窃 电子侵入 不正当手段 未经软件著作权人许可 发表或者登记",
    "偷我的项目": "侵犯商业秘密 盗窃 未经软件著作权人许可 发表或者登记",
    "机密": "商业秘密 保密措施 披露",
    "公开了": "披露 发表 未经软件著作权人许可",
    # 民生消费补足欺诈、撤销与质量瑕疵，避免“车/事故”等物品词代替真实争议性质。
    **dict.fromkeys(("退一赔三", "假一赔三"),
                    "提供商品或者服务有欺诈行为 增加赔偿其受到的损失 购买商品的价款或者接受服务的费用的三倍"),
    "泡水车": "提供商品或者服务有欺诈行为 以欺诈手段 受欺诈方有权请求人民法院或者仲裁机构予以撤销 标的物不符合质量要求",
    "事故车": "提供商品或者服务有欺诈行为 以欺诈手段 受欺诈方有权请求人民法院或者仲裁机构予以撤销",
    **dict.fromkeys(("霸王条款", "概不负责", "概不管", "霸王合同", "统一印的", "格式合同"),
                    "格式条款 不合理地免除或者减轻其责任 排除或者限制消费者权利 店堂告示 声明 "
                    "提供格式条款一方不合理地免除或者减轻其责任、加重对方责任、限制对方主要权利 "
                    "提供格式条款一方排除对方主要权利 该格式条款无效"),
    # 预付消费和农村房屋争议补充履行、退款、保护与补偿结构，不据口语认定权属或退款金额。
    **dict.fromkeys(("办卡", "充钱", "充值", "预付卡", "店关了", "卷钱跑路", "跑路了"),
                    "经营者以预收款方式提供商品或者服务的 应当按照约定提供 未按照约定提供的 "
                    "应当按照消费者的要求履行约定或者退回预付款 并应当承担预付款的利息"),
    **dict.fromkeys(("农村老房子", "老房子拆迁", "宅基地拆迁", "城里人不给"),
                    "依法取得的宅基地和宅基地上的农村村民住宅及其附属设施受法律保护 "
                    "禁止违背农村村民意愿强制流转宅基地 禁止违法收回农村村民宅基地 "
                    "征收土地应当给予公平、合理的补偿 遗产是自然人死亡时遗留的个人合法财产"),
    # 宠物受损与动物致人损害是不同路径；同时召回财产损害与刑事线索，不预判定罪。
    "投毒": "投放毒害性 故意毁坏公私财物 数额较大或者有其他严重情节 侵害物权",
    "毒死": "故意毁坏公私财物 投放毒害性 故意损毁公私财物 侵害物权 造成损害的，权利人可以依法请求损害赔偿",
    "高空抛物": "从建筑物中抛掷物品 从建筑物上坠落的物品造成他人损害 难以确定具体侵权人",
    "人肉搜索": "侵害他人隐私权 个人信息受法律保护 网络用户利用网络服务实施侵权行为 停止侵害、消除影响、恢复名誉、赔礼道歉",
    "开盒": "侵害他人隐私权 个人信息受法律保护 非法获取、出售或者提供公民个人信息",
    "造谣": "以侮辱、诽谤或者其他方式侵害他人名誉权 公然侮辱他人或者捏造事实诽谤他人 通知网络服务提供者采取删除、屏蔽、断开链接",
    # 平台明知侵权未删除：只认“我的私照/私照被发”等具体传播表述，
    # 不认裸词“平台/私照”；是否“知道或者应当知道”仍须核对通知与处理事实。
    **dict.fromkeys(("我的私照", "私照被发"),
                    "网络服务提供者知道或者应当知道 未采取必要措施 承担连带责任"),
    **dict.fromkeys(("群里说我", "全是编的", "污蔑我"),
                    "以侮辱、诽谤或者其他方式侵害他人名誉权 公然侮辱他人或者捏造事实诽谤他人 "
                    "通知网络服务提供者采取删除、屏蔽、断开链接 任何组织或者个人不得以侮辱、诽谤等方式侵害他人的名誉权"),
    # 校园领证争议召回学生评价、毕业条件和救济权，不能只由“证书”词吸向撤销假证规则。
    # 不据未参加比赛直接推定符合毕业或授位条件，仍须核对修业成绩、品行及学校具体决定。
    **dict.fromkeys(("毕业证", "不让我毕业", "卡毕业", "扣留毕业证", "不发毕业证"),
                    "在学业成绩和品行上获得公正评价 完成规定的学业后获得相应的学业证书 学位证书 "
                    "对学校、教师侵犯其人身权、财产权等合法权益，提出申诉或者依法提起诉讼 "
                    "思想品德合格，在规定的修业年限内学完规定的课程，成绩合格或者修满相应的学分，准予毕业"),
    **dict.fromkeys(("学位证", "不给学位", "卡学位"),
                    "完成规定的学业后获得相应的学业证书 学位证书 "
                    "对不受理其学位申请、不授予其学位或者撤销其学位等行为不服的，可以向学位授予单位申请复核，或者请求有关机关依照法律规定处理"),
    # 专题扩展同时服务本地BM25与缺失司法解释目录定位，不据口语认定退赔、产权或立案条件。
    **dict.fromkeys(("刷脸", "人脸识别"),
                    "使用人脸识别技术处理个人信息 物业服务企业或者其他建筑物管理人以人脸识别作为出入唯一验证方式 "
                    "不同意通过人脸识别进行验证 合理验证方式"),
    **dict.fromkeys(("拆封不退", "七天无理由", "直播带货", "网购"),
                    "审理网络消费纠纷 消费者因检查商品的必要对商品进行拆封 七日无理由退货 电子商务平台"),
    **dict.fromkeys(("退一赔十", "过期食品", "变质食品", "假药"),
                    "审理食品药品惩罚性赔偿 审理食品药品纠纷 生产不符合食品安全标准的食品或者经营明知是不符合食品安全标准的食品 价款十倍"),
    **dict.fromkeys(("逾期交房", "延期交房", "逾期办证", "一房二卖", "买期房"),
                    "审理商品房买卖合同 商品房消费者权利保护 出卖人迟延交付房屋 迟延办理房屋所有权登记"),
    **dict.fromkeys(("财产保全", "冻结账户", "查封财产"),
                    "办理财产保全案件 人民法院采取保全措施 申请财产保全"),
    **dict.fromkeys(("微信聊天记录", "电子证据", "录音证据"),
                    "民事诉讼证据 电子数据 视听资料 当事人对自己提出的主张"),
    **dict.fromkeys(("立案标准", "多少钱立案", "够判刑吗", "够刑事立案吗"),
                    "办理诈骗刑事案件具体应用法律 办理盗窃刑事案件适用法律 数额较大 数额巨大 三千元至一万元以上 一千元至三千元以上"),
    # 相邻通风、采光和日照妨害：只认具体的挡光、遮挡动作，不把“高墙”这类名词当触发词；
    # 是否超过必要限度仍须核对事实，词条本身不认定妨害成立。
    **dict.fromkeys(("挡光", "遮光", "挡采光", "遮采光", "起高墙", "盖高墙", "建高墙", "高墙遮挡"),
                    "相邻 通风 采光 日照 妨碍"),
    # 行政处罚的从轻、减轻：只认“少罚/减轻/主动消除危害”等诉求表述，
    # 不把普通的“罚钱”问句拉向从轻条款；是否主动改正、赔偿损失仍须核对事实。
    **dict.fromkeys(("少罚", "轻罚", "罚钱能少", "少罚点", "罚少点", "罚太多",
                     "主动消除危害", "主动减轻危害", "从轻减轻", "减轻处罚", "从轻处罚"),
                    "从轻 减轻 行政处罚 危害后果"),
    # 租赁物被查封、扣押致无法使用：只认查封、封门等限制动作，
    # 普通“退租”不触发，避免把正常解约问句拉向查封条款。
    **dict.fromkeys(("封住", "被封", "查封", "不让进", "门被封"),
                    "查封 扣押 无法使用 解除合同"),
    # 撤销学位与复核：只认撤销、收回、不发学位的具体表述，不认泛词“学位”，
    # 避免把学位申请、授权点等问句拉向撤销条款；不预判撤销决定是否合法。
    **dict.fromkeys(("撤销学位", "学位不算", "学位被撤", "收回学位", "撤销了学位", "不发学位"),
                    "撤销 复核 学位授予单位"),
    # 租赁物正常使用产生损耗：补约定使用方法与免责表述，损耗是否“正常”仍看事实。
    **dict.fromkeys(("磨旧", "磨损", "用旧", "正常损耗"),
                    "约定的方法 租赁物 损耗 不承担赔偿责任"),
    # 肖像被他人盗用获利：只认盗发、盗用照片、偷用头像等侵权行为，
    # 不认泛词“照片/赔多少”，避免把证件照、普通索赔问句拉向肖像赔偿条款。
    **dict.fromkeys(("盗图", "盗发", "盗用照片", "盗发照片", "偷用头像", "乱发私照", "盗用我照片"),
                    "人身权益 财产损失 获得的利益 赔偿数额"),
    # 申请行政机关履行法定职责、久拖不办：只认机关/部门拖延与不作为表述，
    # 不认泛词“不理/去告/告状”，避免把普通纠纷问句拉向行政诉讼条款。
    **dict.fromkeys(("不作为", "机关不理", "部门不理", "两三个月不理", "几个月不理",
                     "拖着不理", "拖着不办", "迟迟不理", "不履行法定职责"),
                    "法定职责 不履行 提起诉讼 行政机关"),
}
# 已有“彩礼”保留完整的返还、赠与与撤销词串，只追加此前缺少的生活困难要件。
COLLOQUIAL_ALIASES["彩礼"] += " 婚前给付并导致给付人生活困难"


def expand_colloquial_query(query):
    """不区分大小写匹配问句，返回保留原文并追加去重术语的字符串；无命中则原样返回。

    仅做子串匹配，不识别否定语境或判断法律适用；无文件、网络及会话副作用。
    """
    text_lower = query.lower()
    extras = []
    seen = set()
    # 按空格分隔的完整短语去重：既有词组补充新要件时，共同短语不重复，新要件不丢失。
    # 比较忽略大小写；只整理追加部分，不改变用户原话及首次出现短语的拼写。
    for phrase, terms in COLLOQUIAL_ALIASES.items():
        if phrase.lower() not in text_lower:
            continue
        for term in terms.split():
            key = term.lower()
            if key not in seen:
                extras.append(term)
                seen.add(key)
    if not extras:
        return query
    return query + " " + " ".join(extras)


def tokens(text):
    """生成关键词检索所用词项：中文使用相邻双字组合，英文数字按连续词处理；不是法律语义分析。"""
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", " ", text.lower())
    result = []
    for word in text.split():
        if re.fullmatch(r"[a-z0-9]+", word):
            result.append(word)
        else:
            result.extend(word[i:i + 2] for i in range(len(word) - 1))
    return result


class BM25Index:
    """预计算的BM25内存倒排索引：避免每次提问对4.6万条法规重复分词。"""
    def __init__(self, rows):
        """对传入的条文列表一次性完成双字分词、文档频率与长度统计。"""
        self.row_count = len(rows)
        self.postings = {}
        lengths = []
        for i, r in enumerate(rows):
            doc = Counter(tokens(r.get("text", "")))
            length = sum(doc.values())
            lengths.append(length)
            for term, tf in doc.items():
                self.postings.setdefault(term, []).append((i, tf))
        self.lengths = lengths
        self.avg = sum(lengths) / max(self.row_count, 1) or 1

    def rank(self, query):
        """利用倒排表快速计算查询词项的BM25得分，结果与全量逐行计算完全一致。"""
        terms = set(tokens(query))
        if not terms or not self.row_count:
            return []
        scores = {}
        for term in terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (self.row_count - df + 0.5) / (df + 0.5))
            for doc_idx, tf in posting:
                score = idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * self.lengths[doc_idx] / self.avg))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + score
        return sorted(((i, s) for i, s in scores.items() if s > 0), key=lambda x: (-x[1], x[0]))


def bm25_rank(rows, query):
    """恢复仅正文BM25；不对法名硬加权，不预设基础法或适用性。"""
    return BM25Index(rows).rank(query)


def normalize_normative_clause(text):
    """规范分句比较键：去标点并统一条件分项句末的“的”；不改原文，无外部副作用。"""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text).removesuffix("的")


def fuse_ranks(lexical, dense):
    """按原有两参数等权 RRF 融合行号名次；返回分数降序、同分行号升序列表，无外部副作用。"""
    scores = {}
    for ranking in (lexical, dense):
        for rank, i in enumerate(ranking):
            scores[i] = scores.get(i, 0) + 1 / (30 + rank + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def load_corpus():
    """加载原有规范、案例摘要与新增条文，并检查快照/产物一致性；失败即拒绝该次加载。"""
    from prototype.importer import load_documents, capture_for
    from prototype.verifier import load_raw, verify_article_exact, verify_quote
    rows = []
    docs = load_documents()
    for doc in docs:
        raw = load_raw(doc.doc_id)
        for art in doc.articles:
            ok, reason = verify_article_exact(raw, art.article_no, art.text)
            if not ok:
                raise RuntimeError(f"原文快照核验失败：{doc.doc_id}/{art.article_no}：{reason}")
            rows.append({"id": doc.doc_id + "/" + doc.version_id + "/" + art.article_id,
                         "kind": "LAW_ARTICLE", "title": doc.title, "locator": art.article_no + " " + art.locator,
                         "version_id": doc.version_id, "text": art.text, "source_url": doc.official_url,
                         "content_scope": "FULL_TEXT", "verified_scope": "与已保存原文快照逐字一致；非在线复核，不代表版本适用"})
    cases = list((ROOT / "data/cases").glob("*.json"))
    if cases:
        from prototype.case_importer import RELEASE_FILE
        manifest = json.loads((ROOT / "data/raw/capture_manifest.json").read_text(encoding="utf-8"))
        entry = next(x for x in manifest["captures"] if x["raw_file"] == RELEASE_FILE)
        capture_for(entry["doc_id"], RELEASE_FILE)
        raw = (ROOT / "data/raw" / RELEASE_FILE).read_text(encoding="utf-8")
        for file in sorted(cases):
            case = json.loads(file.read_text(encoding="utf-8"))
            for n, section in enumerate(case["sections"]):
                if not verify_quote(raw, section["text"])[0]:
                    raise RuntimeError("案例发布稿快照核验失败")
                rows.append({"id": case["case_id"] + "/" + str(n), "kind": "CASE_RELEASE_EXCERPT",
                             "title": case["title"] + " / " + section["section_type"],
                             "locator": section["locator"], "text": section["text"],
                             "version_id": case.get("retrieved_at", "未知"), "source_url": case["source_url"],
                             "content_scope": "RELEASE_SUMMARY_ONLY",
                             "verified_scope": "与已保存发布稿快照逐字一致；仅摘要，非判决书全文"})
    from .ingest import load_extra
    extra, records = load_extra()
    rows.extend(extra)
    return rows, {"laws": len(docs)+len(records), "cases": len(cases), "chunks": len(rows),
                  "fields": sorted({r['field'] for r in records} | {'LABOR'}),
                  "note": f"已加载{len(docs)+len(records)}部规范、{len(cases)}个劳动案例摘要；新增官网文本的现行性与适用待复核，尚非全部法律。"}


class Retriever:
    """检索协调器：维护向量缓存、BM25内存索引与入库状态，融合关键词和数据库召回结果。"""
    def __init__(self, rows, embedder, cache_path=None, *, store=None, doc_format=DOC_FORMAT_PLAIN):
        """初始化对象依赖与内部状态；后续读写通过本类的方法完成。

        doc_format 只影响向量化文本与缓存/集合命名；默认 plain 与改造前完全一致。
        非 plain 使用独立缓存文件，避免误读旧格式向量。
        """
        if doc_format not in DOC_FORMATS:
            raise ValueError(f"未知文档向量化格式：{doc_format}")
        self.rows = rows
        self.embedder = embedder
        self.doc_format = doc_format
        default_cache = (ROOT / "data/local_index/embeddings.json" if doc_format == DOC_FORMAT_PLAIN
                         else ROOT / f"data/local_index/embeddings.{doc_format}.json")
        self.cache_path = Path(cache_path or default_cache)
        self.vectors = None
        self.state = "pending"
        self.error = ""
        self.completed = 0
        self.lock = asyncio.Lock()
        self.store = store

    @property
    def rows(self):
        """返回当前检索语料列表。"""
        return self._rows

    @rows.setter
    def rows(self, value):
        """更新语料时同步重建BM25与完整规范分句索引，避免重载后使用旧行号。"""
        self._rows = list(value)
        self._bm25 = BM25Index(self._rows)
        # 全库唯一性核验，不只在本轮候选里找“唯一”；同条重复不算歧义，不同条同句则拒绝保护。
        self._clause_owners = {}
        for i, row in enumerate(self._rows):
            body = re.sub(r"^第[零〇一二三四五六七八九十百千万两\d]+条(?:之[零〇一二三四五六七八九十百千万两\d]+)?\s*", "", row.get("text", ""))
            for clause in re.split(r"[。；;\n]", body):
                clause = re.sub(r"^\s*[（(][零〇一二三四五六七八九十百千万两\d]+[）)]", "", clause)
                if not re.search(r"应当|不得|可以|有权|未", clause):
                    continue
                key = normalize_normative_clause(clause)
                owner = self._clause_owners.get(key, i)
                self._clause_owners[key] = i if owner == i else None

    def _exact_clause_anchors(self, query, eligible, scope_factor):
        """找已有扩展短语完整且唯一对应的规范分句；只保护原合并池内、通过范围门控的候选。

        不凭法名、条号或绝对BM25门槛识别；不把概念标签、部分句或全库多处同句当作精确锚点。
        只读已建内存索引，不联网、不改分数或原始资料。
        """
        anchors = []
        for phrase, terms in COLLOQUIAL_ALIASES.items():
            if phrase.lower() not in query.lower():
                continue
            for term in terms.split():
                i = self._clause_owners.get(normalize_normative_clause(term))
                if i is not None and i in eligible and scope_factor(i) == 1.0 and i not in anchors:
                    anchors.append(i)
        return anchors
    @staticmethod
    def normalize(v):
        """检查向量数值并归一化，拒绝空、零或非有限向量，供余弦相似度计算使用。"""
        if not v or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
            raise ValueError("无效向量")
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            raise ValueError("零向量")
        return [x / norm for x in v]

    def _compound_intent_anchors(self, query, lexical_scores, scope_factor):
        """从已有口语映射分组取最多四个独立强要件锚点，返回行号；只读内存，不调模型/网络。

        分支首名须BM25≥85、领先次名至少25%，且整句词面≥85、通过原范围门控。
        相似词组去重后至少两支才生效；这是候选覆盖保护，不证明多个法律事实同时成立。
        """
        groups = {}
        lowered = query.lower()
        for phrase, terms in COLLOQUIAL_ALIASES.items():
            position = lowered.find(phrase.lower())
            if position >= 0:
                groups[terms] = min(groups.get(terms, position), position)
        anchors, term_sets = [], []
        for terms, _ in sorted(groups.items(), key=lambda item: item[1]):
            terms_set = set(tokens(terms))
            if any(len(terms_set & other) / max(1, len(terms_set | other)) >= .5 for other in term_sets):
                continue
            branch = self._bm25.rank(terms)
            if not branch:
                continue
            index, score = branch[0]
            runner_up = branch[1][1] if len(branch) > 1 else 0.0
            if (score < 85.0 or score < runner_up * 1.25 or lexical_scores.get(index, 0.0) < 85.0
                    or scope_factor(index) != 1.0):
                continue
            term_sets.append(terms_set)
            if index not in anchors:
                anchors.append(index)
            if len(anchors) >= 4:
                break
        return anchors if len(anchors) >= 2 else []

    async def build(self):
        """按模型指纹与条文哈希复用向量，生成缺失项并保存断点；配置数据库时还须完成整批写入。"""
        async with self.lock:
            self.state = "building"
            keys = [hashlib.sha256((r["id"] + "\n" + r["text"]).encode()).hexdigest() for r in self.rows]
            fingerprint = hashlib.sha256(json.dumps([self.embedder.model_id, keys]).encode()).hexdigest()
            if self.doc_format != DOC_FORMAT_PLAIN:
                # 新格式把格式名和实际送模文本一起写进 key 与指纹，绝不复用旧格式向量。
                texts = [document_embedding_text(r, self.doc_format) for r in self.rows]
                keys = [hashlib.sha256((r["id"] + "\n" + self.doc_format + "\n" + text).encode()).hexdigest()
                        for r, text in zip(self.rows, texts)]
                fingerprint = hashlib.sha256(
                    json.dumps([self.embedder.model_id, self.doc_format, keys]).encode()).hexdigest()
            cached = {}
            try:
                cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if cached.get("model_id") != self.embedder.model_id:
                    cached = {}
                elif self.doc_format != DOC_FORMAT_PLAIN and cached.get("doc_format") != self.doc_format:
                    cached = {}
            except (OSError, ValueError, TypeError):
                pass
            entries = cached.get("entries", {})
            if not isinstance(entries, dict): entries = {}
            vectors = []
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            def checkpoint():
                """原子保存当前已完成的资料向量，供失败重试复用；不保存用户问题或未完成向量。"""
                ready = {k: v for k, v in zip(keys, vectors) if v is not None}
                tmp = self.cache_path.with_suffix(".tmp")
                payload = {"fingerprint": fingerprint, "model_id": self.embedder.model_id, "entries": ready}
                if self.doc_format != DOC_FORMAT_PLAIN:
                    payload["doc_format"] = self.doc_format
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                tmp.replace(self.cache_path)
            try:
                for k in keys:
                    try: vectors.append(self.normalize(entries[k]))
                    except (KeyError, ValueError, TypeError): vectors.append(None)
                dimensions = {len(v) for v in vectors if v is not None}
                if len(dimensions)>1: vectors = [None]*len(keys)
                self.completed = sum(v is not None for v in vectors)
                missing = [i for i,v in enumerate(vectors) if v is None]
                for start in range(0, len(missing), 16):
                    indices = missing[start:start+16]
                    new = await self.embedder.embed([document_embedding_text(self.rows[i], self.doc_format)
                                                     for i in indices])
                    if len(new)!=len(indices): raise ValueError("向量数量不符")
                    for i,v in zip(indices,new): vectors[i] = self.normalize(v)
                    self.completed += len(indices)
                    if start % 128 == 0: checkpoint()
                if not vectors or len({len(v) for v in vectors})!=1: raise ValueError("向量维数不一致")
                checkpoint()
                self.vectors = vectors
                if self.store:
                    # 先完成向量缓存，再完整写入数据库；写入失败不宣称数据库已就绪。
                    await self.store.sync(self.rows,vectors,self.embedder.model_id,self.doc_format)
                self.state = "ready"
                self.error = ""
            except Exception:
                checkpoint()
                self.state = "lexical"
                self.error = "Qdrant未就绪或向量未全部入库，已保存向量缓存，暂用关键词检索；可重新检查索引" if self.store else "向量未全部入库，已保存进度，暂用关键词检索；可重新构建继续"
                return

    async def search(self, query, top_k=4):
        """检索当前资料：两路各粗召回20候选，再以救济结构、词面强度和多法名覆盖精排。

        输入问句与 Top-K，返回候选字典列表及模式；可能调用向量网络服务，不写查询缓存。
        单次语义查询失败只更新本轮降级状态，不改数据库入库标志；后续查询成功自动清除错误。
        高置信度原问句前两名可补池外词面共识；五字原句锚点不使用口语扩展内容。
        无合格向量时保留原始词面排序；法名和阈值只是启发式，不认证案由或法律适用。
        """
        if top_k <= 0:
            return [], 'lexical'
        expanded = expand_colloquial_query(query)
        lexical_scores = self._bm25.rank(expanded)
        lexical = [i for i, _ in lexical_scores[:CANDIDATE_LIMIT]]
        dense = []
        mode = "lexical"
        if self.vectors:
            try:
                async with asyncio.timeout(45):
                    query_vec = self.normalize((await self.embedder.embed([query], query=True))[0])
                if len(query_vec) != len(self.vectors[0]):
                    raise ValueError("向量维数不符")
                if self.store:
                    similarities = await self.store.search(query_vec,CANDIDATE_LIMIT)
                else:
                    similarities = [(i, sum(x * y for x, y in zip(query_vec, v))) for i, v in enumerate(self.vectors)]
                dense = [i for i, s in sorted(similarities, key=lambda x: -x[1])[:CANDIDATE_LIMIT] if s >= 0.38]
                mode = "hybrid_qdrant" if self.store else "hybrid"
                self.state = "ready"
                self.error = ""
            except Exception:
                mode = "lexical_qdrant_unavailable" if self.store else "lexical_fallback"
                # ready 表示当前批次已完成入库校验，网络抖动不能抹掉它，否则后续查询永远无法重试。
                # 真正未入库时仍由 store.search() 拒绝查询，本分支不会把 ready 改为 True。
                self.state = "lexical"
                self.error = ("语义查询暂不可用，本轮已降级为关键词检索；后续提问会重新尝试，未完成入库时仍需检查索引"
                              if self.store else "向量查询暂不可用，本轮已降级为关键词检索；后续提问会重新尝试")
        if dense:
            dense_set = set(dense)
            query_lower = query.lower()

            def scope_factor(index):
                """按原问句的主体/工程线索调节候选，不让扩展词反过来放行专门法规。"""
                title = self.rows[index].get("title") or ""
                for title_keys, query_keys in SPECIAL_SCOPE_RULES:
                    if (any(key in title for key in title_keys)
                            and not any(key in query_lower for key in query_keys)):
                        return 0.4
                return 1.0

            # 缺失法名或范围线索不足的专门法不能为同法候选提供语义印证。
            dense_titles = {self.rows[i]["title"] for i in dense[:6]
                            if self.rows[i].get("title") and scope_factor(i) == 1.0}
            lexical_score_map = dict(lexical_scores)
            lexical_rank_map = {i: rank for rank, (i, _) in enumerate(lexical_scores)}
            dense_similarity_map = dict(similarities)
            original_fragments = {segment[start:start+5] for segment in re.findall(r"[\u4e00-\u9fff]{5,}", query)
                                  for start in range(len(segment)-4)}
            boosted_title_counts = {}
            # 开关只看原问句，不能让扩展词中的“有关机关”等表述反过来改变用户查询意图。
            adjust_remedy_intent = not any(keyword in query for keyword in OFFICIAL_QUERY_KEYWORDS)
            # 超强词面可超过单路语义，但每部法最多两条享受提权；其余按共识或强度保留/衰减。
            # 本分支独立累加，原等权 fuse_ranks() 的两参数接口与行为保持不变。
            scores = {}
            # 同类量刑结构在相邻语义候选中重复出现时，保留语义路自身提供的结构支持。
            # 这样不把刑事问题中成组召回的条文与普通纠纷里孤立混入的量刑句一并降权。
            sanction_ranks = {rank for rank, i in enumerate(dense)
                              if scope_factor(i) == 1.0 and DIRECT_SANCTION_RE.search(self.rows[i].get("text", ""))}
            for rank, i in enumerate(dense):
                intent_factor = 1.0
                if adjust_remedy_intent:
                    text = self.rows[i].get("text", "")
                    if any(marker in text for marker in MISMATCHED_OFFICIAL_MARKERS):
                        intent_factor = 0.55
                    elif ((any(marker in text for marker in REMEDY_STRUCT_MARKERS)
                           or expanded == query and (PROTECTIVE_STRUCT_RE.search(text)
                               or DIRECT_SANCTION_RE.search(text)
                               and (rank < 2 or lexical_score_map.get(i, 0.0) >= 15.0
                                    or rank - 1 in sanction_ranks or rank + 1 in sanction_ranks)))
                          and not (GENERIC_LIABILITY_RE.search(text) and lexical_score_map.get(i, 0.0) < 35.0)):
                        # 新增保护/制裁结构只补原句未扩展的盲区；已有法定锚点时避免泛制裁覆盖精确要件。
                        # 沿用35分词面支持界限：通用责任套话不能仅凭“承担民事责任”压过更靠前的语义候选。
                        # 只取消这类候选的额外35%奖励，原语义贡献、范围门控和召回资格都保留。
                        intent_factor = 1.35
                scores[i] = scores.get(i, 0.0) + (1.2 * intent_factor * scope_factor(i)) / (30 + rank + 1)
                # 候选池截断只限制召回，不应抹掉池内向量候选已有的词面证据。
                # 仅对未扩展原句、相似度≥0.65的前两名且BM25≥15补分；避免给扩展长句的泛匹配加分。
                # 不引入新候选；按真实排名衰减，范围门控仍生效。这些是评测校准的启发式阈值。
                lexical_rank = lexical_rank_map.get(i)
                if (expanded == query and rank < 2 and dense_similarity_map.get(i, 0.0) >= 0.65
                        and lexical_score_map.get(i, 0.0) >= 15.0
                        and lexical_rank is not None and lexical_rank >= CANDIDATE_LIMIT):
                    scope = scope_factor(i)
                    weight = 1.0 if scope == 1.0 else 0.25 * scope
                    scores[i] += weight / (30 + lexical_rank + 1)
            for rank, i in enumerate(lexical):
                title = self.rows[i].get("title")
                lexical_score = lexical_score_map.get(i, 0.0)
                scope = scope_factor(i)
                # 范围线索不足时，双路重合和高 BM25 都不能绕过门控，也不触发多法名保护。
                if scope != 1.0:
                    weight = 0.25 * scope
                elif lexical_score >= 85.0 and title and boosted_title_counts.get(title, 0) < 2:
                    # 多短语精确匹配可压过单路救济向量1.62；85≤分数<140仍为1.35，共用两条提权名额。
                    weight = 2.1 if lexical_score >= 140.0 else 1.35
                    boosted_title_counts[title] = boosted_title_counts.get(title, 0) + 1
                elif i in dense_set or (title and title in dense_titles) or lexical_score >= 35.0:
                    weight = 1.0
                    # 原问句连续五字要件与词面首名精确吻合时，补偿单向量遗漏；扩展词不能触发。
                    # 只作用于35≤分数<85的首名，保留既有85/140锚点分档和短碎片边界。
                    if (rank == 0 and 35.0 <= lexical_score < 85.0
                            and any(fragment in self.rows[i].get("text", "") for fragment in original_fragments)):
                        weight = 1.65
                else:
                    weight = 0.25
                scores[i] = scores.get(i, 0.0) + weight / (30 + rank + 1)
            ranked_all = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
            if len(boosted_title_counts) >= 2:
                # 多法名超强锚点提示复合问题，首轮每法至多两条；不删候选，超额部分保序回填。
                # 这只是法名多样性保护，不等同于案由判定；单法问题仍保留原融合次序。
                selected, overflow = [], []
                selected_title_counts = {}
                for i, score in ranked_all:
                    title = self.rows[i].get("title")
                    if title and selected_title_counts.get(title, 0) >= 2:
                        overflow.append((i, score))
                    else:
                        selected.append((i, score))
                        if title:
                            selected_title_counts[title] = selected_title_counts.get(title, 0) + 1
                ranked = selected + overflow
            else:
                ranked = ranked_all
            if top_k >= 2:
                # 完整唯一分句是短条文的直接词面证据，不受正文长度造成的85分绝对门槛限制。
                anchors = list(dict.fromkeys(self._exact_clause_anchors(query, set(scores), scope_factor)
                                             + self._compound_intent_anchors(query, lexical_score_map, scope_factor)))[:top_k]
                if anchors and not set(anchors) <= {i for i, _ in ranked[:top_k]}:
                    # 只在确实遗漏独立强分支时保留锚点，其余名额仍按原融合次序；不按法名写死目标。
                    known = {i for i, _ in ranked}
                    ranked += [(i, 1.35 / (31 + lexical_rank_map[i])) for i in anchors if i not in known]
                    selected = set(anchors)
                    for i, _ in ranked:
                        if len(selected) >= top_k:
                            break
                        selected.add(i)
                    ranked = [item for item in ranked if item[0] in selected] + [item for item in ranked if item[0] not in selected]
        else:
            # 无合格向量候选时，连原始 BM25 分值也保持不变，不把降级结果换算成 RRF。
            ranked = lexical_scores
        # 不按标题折叠，按本次融合或词面排序保留同法多核心条款；Top-K为零仍明确返回空。
        return [dict(self.rows[i], retrieval_score=round(score, 5)) for i, score in ranked[:top_k]], mode

    async def tail_candidates(self, query, exclude_ids, depth=TAIL_DEPTH):
        """取页面"更多候选"：两路各自全库前 depth 名里、不在 exclude_ids 中的条文，按最好名次排序。

        关键词用扩展后的问题、语义用原问题且仍卡0.38，与线上两路口径一致；每条带 tail_routes 记两路名次。
        只供展示、不送模型；可能调用向量服务，不写缓存。语义查询失败时只返回关键词部分，
        且不改数据库就绪状态，避免一次展示失败变成后续一直降级。
        """
        bm25 = [i for i, _ in self._bm25.rank(expand_colloquial_query(query))[:depth]]
        dense = []
        if self.vectors:
            try:
                async with asyncio.timeout(45):
                    query_vec = self.normalize((await self.embedder.embed([query], query=True))[0])
                if self.store:
                    similarities = await self.store.search(query_vec, depth)
                else:
                    similarities = [(i, sum(x * y for x, y in zip(query_vec, v))) for i, v in enumerate(self.vectors)]
                dense = [i for i, s in sorted(similarities, key=lambda x: -x[1])[:depth] if s >= 0.38]
            except Exception:
                dense = []
        ranks = {}
        for route, ranking in (("bm25", bm25), ("dense", dense)):
            for rank, i in enumerate(ranking, 1):
                ranks.setdefault(i, {"bm25": None, "dense": None})[route] = rank

        def order(item):
            i, routes = item
            best = min(rank for rank in routes.values() if rank is not None)
            # 名次相同时语义在前：实测口语问题里语义一路更可靠，见 docs/guides/RETRIEVAL_MEASUREMENT.md。
            return best, 0 if routes["dense"] == best else 1, i

        return [dict(self.rows[i], tail_routes=routes) for i, routes in sorted(ranks.items(), key=order)
                if self.rows[i].get("id") not in exclude_ids]
