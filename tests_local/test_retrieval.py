"""检索核心逻辑与向量缓存机制自动化测试

用途说明：
验证 local_rag/retrieval.py 中的检索与缓存机制，包含：
1. BM25 词面打分能力：精准法律术语的权重高于泛化高频词。
2. Top-20粗召回、救济结构精排、RRF默认兼容、85/140分锚点与多法名保序限流及降级边界。
3. 全局语料加载：验证加载的条文与案例摘要具有唯一 ID 和真实性哈希标记。
4. 向量缓存性能与隐私保护：
   - 资料向量只生成一次并存盘复用，避免重启重复计算；
   - 用户提问的检索向量【绝对不落盘缓存】，防止用户隐私留痕；
   - 增量入库：新增一条法规时，仅对新法规计算向量，旧法规完全复用缓存；
   - 断点续传：向量计算中途断电或中断，已完成的批次会自动保留，下次从断点继续。
5. 口语扩展与召回：大小写与跨组去重、真实语料校园领证/民生/复合技术/婚恋 Top-4、原问句向量化及 Top-K 边界。
6. 专门主体/工程法规范围门控，以及预付消费、农村房屋和酒店隐私的候选重排覆盖。
7. 第二轮旅游购物、未成年人消费、房屋隐瞒与装修停工Top-4及出国/国企范围边界。
8. 隐晦侵害/被指控的双路径Top-4与完整短语跨组去重，不预设受害人性别或犯罪成立。
"""
import asyncio
import json
import math
from pathlib import Path

import pytest

from local_rag.retrieval import (
    DOC_FORMAT_LAW_HEADER_V1, DOC_FORMAT_PLAIN, ROOT, Retriever, bm25_rank, document_embedding_text,
    expand_colloquial_query, load_corpus, fuse_ranks,
)

COMPOSITE_QUERY = "我想知道就是我做的项目我用的是VPNcodex来做的，然后我的同事窃取我的项目，但是那里面有一点机密的东西我忘记删掉了，然后我同事拿去了然后还公开了之后才知道里面有机密，这种情况怎么办的呢"


class Embedder:
    """测试用向量化桩（Mock Embedder）：模拟 Ollama 的向量计算，记录调用次数以便验证缓存复用"""
    model_id = "fake-embedding:1"

    def __init__(self):
        self.calls = []

    async def embed(self, texts, *, query=False):
        # 记录调用参数：(文本列表, 是否为用户提问)
        self.calls.append((list(texts), query))
        # 根据是否包含“工伤”返回简单的模拟二维向量
        return [[1.0, 0.0] if "工伤" in x else [0.0, 1.0] for x in texts]


def test_dense_candidate_keeps_lexical_consensus_beyond_candidate_cutoff(monkeypatch):
    """双路共识不应由词面Top-20截断偶然抹去；不新增候选或改变纯词面降级。"""
    rows = [dict(id=str(i), title=f"规范{i}", text="一般规则") for i in range(30)]
    retriever = Retriever(rows, Embedder())
    retriever.vectors = [[0.0, 1.0]]
    class Store:
        async def search(self, vector, limit):
            return [(29, .8)]
    retriever.store = Store()
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(i, 25.0) for i in range(30)])
    hits, _ = asyncio.run(retriever.search("没有扩展的口语提问"))
    assert hits[0]["id"] == "29"
    assert hits[0]["retrieval_score"] == round(1.2 / 31 + 1.0 / 60, 5)


def test_original_multi_term_phrase_can_anchor_moderate_lexical_first(monkeypatch):
    """原问句连续五字以上的精确匹配可补足语义缺席；扩展词和碎片词不能独自触发。"""
    rows = [dict(id="anchor", title="合同规范", text="应当采取补救措施或者赔偿损失。"),
            dict(id="dense", title="工程规范", text="有权请求工程价款")]
    retriever = Retriever(rows, Embedder())
    retriever.vectors = [[0., 1.]]
    class Store:
        async def search(self, vector, limit):
            return [(1, .8)]
    retriever.store = Store()
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(0, 59.0)])
    hits, _ = asyncio.run(retriever.search("收了钱不做，我能要求采取补救措施或者赔偿吗"))
    assert hits[0]["id"] == "anchor"
    assert hits[0]["retrieval_score"] == round(1.65 / 31, 5)


@pytest.mark.parametrize("text", ["自然人享有隐私权。", "任何组织或者个人不得拍摄私密空间。",
                                  "盗窃财物的，处三年以下有期徒刑。"])
def test_protective_and_direct_sanction_structures_are_not_ignored(monkeypatch, text):
    """保护性禁止与直接制裁也是规范结构，不能只奖励带赔偿/申诉字样的条文。"""
    rows = [dict(id="other", title="其他规范", text="主管单位组织开展一般工作。"),
            dict(id="core", title="核心规范", text=text)]
    retriever = Retriever(rows, Embedder())
    retriever.vectors = [[0., 1.]]
    class Store:
        async def search(self, vector, limit):
            return [(0, .8), (1, .75)]
    retriever.store = Store()
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [])
    hits, _ = asyncio.run(retriever.search("遇到这个事情怎么办"))
    assert hits[0]["id"] == "core"
    assert hits[0]["retrieval_score"] == round(1.2 * 1.35 / 32, 5)


@pytest.mark.parametrize('rank, lexical_score, neighbor, factor', [(1, 0, False, 1.35), (5, 0, False, 1.0), (5, 14.999, False, 1.0), (5, 15.0, False, 1.35), (5, 0, True, 1.35)])
def test_direct_sanction_boost_requires_early_dense_rank_or_lexical_support(monkeypatch, rank, lexical_score, neighbor, factor):
    """量刑结构须有前列、词面或相邻同类语义支持，不奖励中后段孤立罪名。"""
    rows = [dict(id=str(i), title=f'匿名规范{i}', text='普通说明。') for i in range(7)]
    rows[6]['text'] = '实施某行为的，处三年以下有期徒刑。'
    retriever = Retriever(rows, Embedder())
    retriever.vectors = [[0., 1.]]
    order = list(range(6))
    order.insert(rank, 6)
    if neighbor:
        rows[order[rank - 1]]['text'] = '实施另一行为的，处三年以下有期徒刑。'
    class Store:
        async def search(self, vector, limit):
            return [(i, .8) for i in order]
    retriever.store = Store()
    monkeypatch.setattr(retriever._bm25, 'rank', lambda query: [(6, lexical_score)] if lexical_score else [])
    hits, _ = asyncio.run(retriever.search('遇到这种事怎么办', top_k=7))
    score = next(h['retrieval_score'] for h in hits if h['id'] == '6')
    assert score == round(1.2 * factor / (31 + rank) + (1 / 31 if lexical_score else 0), 5)


@pytest.mark.parametrize('duplicate, extra_clause, protected', [(False, False, True), (True, False, False), (False, True, False)])
def test_unique_complete_normative_alias_clause_preserves_pool_candidate(monkeypatch, duplicate, extra_clause, protected):
    """唯一完整规范分句可保护短条文；同句多处出现或只匹配分句的一部分时不保护。"""
    from local_rag.retrieval import COLLOQUIAL_ALIASES
    monkeypatch.setitem(COLLOQUIAL_ALIASES, '约定变动', '当事人可以撤销约定')
    rows = [dict(id=str(i), title=f'匿名规范{i}', text='一般说明。') for i in range(6)]
    rows[5]['text'] = '第一条 当事人可以撤销约定' + ('，但是存在例外' if extra_clause else '') + '。'
    if duplicate:
        rows.append(dict(id='duplicate', title='另一匿名规范', text='第二条 当事人可以撤销约定。'))
    retriever = Retriever(rows, Embedder())
    retriever.vectors = [[0., 1.]]
    class Store:
        async def search(self, vector, limit):
            return [(i, .8) for i in range(5)]
    retriever.store = Store()
    monkeypatch.setattr(retriever._bm25, 'rank', lambda query: [(i, 20.) for i in range(6)])
    hits, _ = asyncio.run(retriever.search('约定变动怎么办', top_k=4))
    assert ('5' in {h['id'] for h in hits}) is protected
    if protected:
        # 实际重载语料后，完整分句移到另一行，不能继续保护旧索引行号。
        changed = [dict(row) for row in rows]
        changed[4]['text'], changed[5]['text'] = changed[5]['text'], '普通说明。'
        retriever.rows = changed
        refreshed, _ = asyncio.run(retriever.search('约定变动怎么办', top_k=4))
        assert '4' in {h['id'] for h in refreshed} and '5' not in {h['id'] for h in refreshed}


def test_exact_legal_terms_beat_generic_words():
    """测试 BM25 关键词匹配：包含具体法律概念（如“工伤保险”）的条文得分必须高于只含泛词（如“用人单位”）的条文"""
    rows = [
        {"id": "a", "text": "用人单位劳动者劳动合同"},
        {"id": "b", "text": "未参加工伤保险由用人单位支付工伤待遇"}
    ]
    # 问“没交工伤保险怎么办”，包含工伤待遇的条文 b (下标 1) 必须排在第 1 位
    assert bm25_rank(rows, "没交工伤保险怎么办")[0][0] == 1


@pytest.mark.parametrize("phrase, terms", [
    ("拖欠工资", ("未及时足额支付劳动报酬", "解除劳动合同", "经济补偿")),
    ("不发工资", ("未及时足额支付劳动报酬", "解除劳动合同")),
    ("克扣工资", ("未及时足额支付劳动报酬", "解除劳动合同")),
    ("不交社保", ("未依法为劳动者缴纳社会保险费", "解除劳动合同")),
    ("没交社保", ("未依法为劳动者缴纳社会保险费", "解除劳动合同")),
    ("不缴社保", ("未依法为劳动者缴纳社会保险费", "解除劳动合同")),
    ("没缴社保", ("未依法为劳动者缴纳社会保险费", "解除劳动合同")),
    ("不买社保", ("未依法为劳动者缴纳社会保险费", "解除劳动合同")),
    ("没买社保", ("未依法为劳动者缴纳社会保险费", "解除劳动合同")),
    ("不按合同办", ("不履行合同义务", "违约责任", "继续履行", "赔偿损失")),
    ("货不对板", ("质量不符合约定", "修理", "更换", "退货")),
    ("交了钱不发货", ("迟延履行", "交付标的物", "违约责任")),
    ("被打伤", ("侵害他人人身权益", "医疗费", "误工减少的收入")),
    ("被狗咬", ("饲养的动物造成他人损害", "侵权责任")),
    ("网上造谣", ("诽谤", "名誉权", "停止侵害", "消除影响")),
    ("闹离婚", ("感情确已破裂", "调解无效", "准予离婚")),
    ("家暴", ("家庭暴力", "人身安全保护令", "实施家庭暴力", "申请人身安全保护令", "离婚损害赔偿")),
    ("离婚分财产", ("夫妻共同财产", "协议不成", "判决")),
    ("不给抚养费", ("不直接抚养子女", "抚养费")),
    ("老人去世房子归谁", ("法定继承", "配偶", "子女", "父母")),
    ("没留遗嘱", ("法定继承", "继承顺序")),
    ("骗了", ("诈骗公私财物", "受欺诈方有权请求人民法院或者仲裁机构予以撤销", "行为人因该行为取得的财产 应当予以返还")),
    ("被骗", ("诈骗公私财物", "受欺诈方有权请求人民法院或者仲裁机构予以撤销", "行为人因该行为取得的财产 应当予以返还")),
    ("对象", ("当事人请求返还按照习俗给付的彩礼", "赠与可以附义务", "民事法律行为无效 被撤销或者确定不发生效力后")),
    ("男朋友", ("当事人请求返还按照习俗给付的彩礼", "赠与附义务的", "应当予以返还")),
    ("女朋友", ("当事人请求返还按照习俗给付的彩礼", "赠与附义务的", "应当予以返还")),
    ("前任", ("当事人请求返还按照习俗给付的彩礼", "赠与附义务的", "应当予以返还")),
    ("花了我", ("当事人请求返还按照习俗给付的彩礼", "赠与附义务的", "应当予以返还")),
    ("彩礼", ("当事人请求返还按照习俗给付的彩礼", "赠与附义务的", "应当予以返还", "婚前给付并导致给付人生活困难")),
    ("分手", ("当事人请求返还按照习俗给付的彩礼", "赠与附义务的", "应当予以返还")),
    ("vpn", ("计算机信息网络直接进行国际联网", "国际出入口信道", "自行建立或者使用其他信道")),
    ("窃取我的项目", ("侵犯商业秘密", "盗窃", "电子侵入", "不正当手段", "未经软件著作权人许可", "发表或者登记")),
    ("偷我的项目", ("侵犯商业秘密", "盗窃", "未经软件著作权人许可", "发表或者登记")),
    ("机密", ("商业秘密", "保密措施", "披露")),
    ("公开了", ("披露", "发表", "未经软件著作权人许可")),
    ("退一赔三", ("提供商品或者服务有欺诈行为", "增加赔偿其受到的损失", "购买商品的价款或者接受服务的费用的三倍")),
    ("假一赔三", ("提供商品或者服务有欺诈行为", "增加赔偿其受到的损失", "购买商品的价款或者接受服务的费用的三倍")),
    ("泡水车", ("提供商品或者服务有欺诈行为", "以欺诈手段", "受欺诈方有权请求人民法院或者仲裁机构予以撤销", "标的物不符合质量要求")),
    ("事故车", ("提供商品或者服务有欺诈行为", "以欺诈手段", "受欺诈方有权请求人民法院或者仲裁机构予以撤销")),
    ("霸王条款", ("格式条款", "不合理地免除或者减轻其责任", "排除或者限制消费者权利")),
    ("投毒", ("投放毒害性", "故意毁坏公私财物", "数额较大或者有其他严重情节", "侵害物权")),
    ("毒死", ("故意毁坏公私财物", "投放毒害性", "故意损毁公私财物", "侵害物权", "造成损害的，权利人可以依法请求损害赔偿")),
    ("高空抛物", ("从建筑物中抛掷物品", "从建筑物上坠落的物品造成他人损害", "难以确定具体侵权人")),
    ("人肉搜索", ("侵害他人隐私权", "个人信息受法律保护", "网络用户利用网络服务实施侵权行为", "停止侵害、消除影响、恢复名誉、赔礼道歉")),
    ("开盒", ("侵害他人隐私权", "个人信息受法律保护", "非法获取、出售或者提供公民个人信息")),
    ("造谣", ("以侮辱、诽谤或者其他方式侵害他人名誉权", "公然侮辱他人或者捏造事实诽谤他人", "通知网络服务提供者采取删除、屏蔽、断开链接")),
    *[(phrase, ("经营者以预收款方式提供商品或者服务的", "应当按照消费者的要求履行约定或者退回预付款", "并应当承担预付款的利息"))
      for phrase in ("办卡", "充钱", "充值", "预付卡", "店关了", "卷钱跑路", "跑路了")],
    *[(phrase, ("依法取得的宅基地和宅基地上的农村村民住宅及其附属设施受法律保护", "禁止违背农村村民意愿强制流转宅基地",
                "禁止违法收回农村村民宅基地", "征收土地应当给予公平、合理的补偿", "遗产是自然人死亡时遗留的个人合法财产"))
      for phrase in ("农村老房子", "老房子拆迁", "宅基地拆迁", "城里人不给")],
])
def test_colloquial_query_adds_statutory_terms(phrase, terms):
    """缺失或误配别名时失败；口语仍保留在开头，法定术语仅作为召回提示。"""
    query = phrase + "怎么办？"
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + " ")
    assert all(term in expanded for term in terms)


@pytest.mark.parametrize("phrase", ["毕业证", "不让我毕业", "卡毕业", "扣留毕业证", "不发毕业证"])
def test_graduation_aliases_add_fair_evaluation_completion_and_appeal_terms(phrase):
    """领证争议必须扩展到公正评价、完成学业及申诉，而不只靠“证书”词面碰撞。"""
    query = "老师" + phrase + "怎么办"
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + " ")
    for term in ("在学业成绩和品行上获得公正评价", "完成规定的学业后获得相应的学业证书", "学位证书",
                 "对学校、教师侵犯其人身权、财产权等合法权益，提出申诉或者依法提起诉讼",
                 "思想品德合格，在规定的修业年限内学完规定的课程，成绩合格或者修满相应的学分，准予毕业"):
        assert term in expanded


@pytest.mark.parametrize("phrase", ["学位证", "不给学位", "卡学位"])
def test_degree_aliases_add_review_and_legal_remedy_terms(phrase):
    """学位拒授类口语须扩展到复核救济，而不是只召回不当授予后的撤销规则。"""
    expanded = expand_colloquial_query("学校" + phrase + "怎么办")
    assert "完成规定的学业后获得相应的学业证书" in expanded and "学位证书" in expanded
    assert "对不受理其学位申请、不授予其学位或者撤销其学位等行为不服的，可以向学位授予单位申请复核，或者请求有关机关依照法律规定处理" in expanded


@pytest.mark.parametrize("phrase, terms", [
    *[(phrase, ("八周岁", "限制民事行为能力", "法定代理人同意、追认"))
      for phrase in ("十岁", "八岁", "九岁", "未成年", "小孩", "熊孩子")],
    *[(phrase, ("八周岁以上的未成年人为限制民事行为能力人", "法定代理人同意、追认",
                "与其年龄、智力、精神健康状况相适应的民事法律行为有效", "应当予以返还", "网络游戏、网络直播",
                "不得向未成年人提供与其民事行为能力不符的付费服务"))
      for phrase in ("刷礼物", "直播打赏", "给主播刷", "游戏充值", "游戏充钱", "直播")],
    *[(phrase, ("受欺诈方有权请求人民法院或者仲裁机构予以撤销", "故意隐瞒与订立合同有关的重要事实或者提供虚假情况",
                "中介人应当就有关订立合同的事项向委托人如实报告", "不得请求支付报酬并应当承担赔偿责任"))
      for phrase in ("死过人", "凶宅", "隐瞒", "中介没说", "房东没说")],
    *[(phrase, ("当事人协商一致，可以变更合同", "当事人一方不履行合同义务或者履行合同义务不符合约定的",
                "继续履行、采取补救措施或者赔偿损失等违约责任"))
      for phrase in ("半路加价", "坐地起价", "加钱才继续干", "不给就停工", "全包合同")],
    *[(phrase, ("不得指定具体购物场所", "不得安排另行付费旅游项目", "不得欺骗、胁迫旅游者消费",
                "旅游者有权在旅游行程结束后三十日内", "办理退货并先行垫付退货货款"))
      for phrase in ("强制购物", "逼我们买", "低价旅游", "低价团", "赶下大巴", "甩客")],
])
def test_second_round_scene_aliases_supply_statutory_elements(phrase, terms):
    """每个新增口语入口都须补齐其法定结构；不能只靠同组另一个触发词偶然召回。"""
    query = phrase + "应该怎么处理"
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + " ")
    assert all(term in expanded for term in terms)


@pytest.mark.parametrize("phrase", ["十岁", "八岁", "九岁", "未成年", "小孩", "熊孩子"])
def test_age_alone_does_not_expand_to_online_payment(phrase):
    """身份与消费行为分开：仅有年龄/称谓时不得凭空追加网络付费或消费返还场景。"""
    expanded = expand_colloquial_query(phrase + "的事情如何处理")
    assert "法定代理人同意、追认" in expanded
    assert not any(term in expanded for term in ("网络游戏", "网络直播", "付费服务", "应当予以返还"))


@pytest.mark.parametrize("phrase, terms", [
    *[(phrase, ("捏造事实诬告陷害他人，意图使他人受刑事追究，情节严重的",
                "不是有意诬陷，而是错告，或者检举失实的",
                "捏造事实诬告陷害他人，企图使他人受到刑事追究或者受到治安管理处罚的",
                "收集能够证实犯罪嫌疑人、被告人有罪或者无罪、犯罪情节轻重的各种证据"))
      for phrase in ("告我强奸", "告我猥亵", "诬告", "陷害", "栽赃", "冤枉我", "反咬一口", "仙人跳")],
    *[(phrase, ("强制猥亵他人或者侮辱妇女的",
                "聚众或者在公共场所当众犯前款罪的", "猥亵他人的"))
      for phrase in ("把我给那个了", "强迫我发生", "被强暴", "被性侵", "强制猥亵", "猥亵我")],
])
def test_assault_and_false_accusation_aliases_supply_legal_elements(phrase, terms):
    """逐个口语入口验证强制猥亵或诬告/无罪取证线索，不据扩展词认定犯罪。"""
    query = phrase + "怎么办"
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + " ")
    assert all(term in expanded for term in terms)
    assert "以暴力、胁迫或者其他方法" not in expanded


def test_combined_assault_and_accusation_aliases_preserve_distinct_full_phrases_once():
    """两组多入口同时命中仍只追加一次完整短语；相似前缀的刑事与治安要件不能误删。"""
    query = "把我给那个了，被性侵猥亵我后还反咬一口告我强奸，诬告陷害栽赃冤枉我，造谣怎么办"
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + " ")
    terms = expanded[len(query):].split()
    assert len(terms) == len(set(terms))
    for term in ("强制猥亵他人或者侮辱妇女的",
                 "捏造事实诬告陷害他人，意图使他人受刑事追究，情节严重的",
                 "不是有意诬陷，而是错告，或者检举失实的",
                 "捏造事实诬告陷害他人，企图使他人受到刑事追究或者受到治安管理处罚的",
                 "公然侮辱他人或者捏造事实诽谤他人"):
        assert terms.count(term) == 1


@pytest.mark.parametrize('phrase, terms', [
    *[(p, ('当事人协商一致，可以变更合同', '违约责任'))
      for p in ('装修全包', '不加就停工', '中途加价', '临时加价')],
    *[(p, ('未及时足额支付劳动报酬', '解除劳动合同', '经济补偿'))
      for p in ('不给工资', '不结工资', '不给加班费', '不结加班', '拖欠加班费')],
    *[(p, ('强制猥亵他人或者侮辱妇女的', '猥亵他人的'))
      for p in ('乱摸', '非礼', '摸我')],
    *[(p, ('捏造事实诬告陷害他人', '不是有意诬陷，而是错告', '有罪或者无罪'))
      for p in ('冤死', '假报案', '编造报案')],
    *[(p, ('格式条款', '不合理地免除或者减轻其责任', '店堂告示', '声明'))
      for p in ('概不负责', '概不管', '霸王合同', '霸王条款')],
])
def test_existing_scene_short_synonyms_expand_without_losing_original(phrase, terms):
    """通用短词能进入已有场景；保留原问句，不把扩展词当作已经成立的事实。"""
    query = phrase + '怎么办'
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + ' ')
    assert all(term in expanded for term in terms)


def test_neutral_exemption_term_alone_does_not_assert_unfairness():
    """中性的免责条款称谓不能单独触发不合理免责场景；提示说明问题也可能用到此词。"""
    query = '免责条款是什么意思'
    assert expand_colloquial_query(query) == query


@pytest.mark.parametrize('phrase, terms', [
    *[(p, ('迟延履行主要债务', '经催告后在合理期限内仍未履行', '当事人可以解除合同'))
      for p in ('一直不送', '催好多次', '迟迟不发')],
    *[(p, ('合同解除后', '终止履行', '请求恢复原状或者采取其他补救措施'))
      for p in ('取消装修', '取消合同', '半途取消')],
    *[(p, ('不履行合同义务', '采取补救措施或者赔偿损失'))
      for p in ('换了牌子', '偷换材料')],
    *[(p, ('提供格式条款一方不合理地免除或者减轻其责任', '加重对方责任', '该格式条款无效'))
      for p in ('霸王条款', '概不负责', '概不管', '霸王合同', '统一印的', '格式合同')],
    *[(p, ('当事人有权进行陈述和申辩', '不得因当事人陈述、申辩而给予更重的处罚'))
      for p in ('不服申辩', '越解释越罚')],
    *[(p, ('申请行政许可', '在法定期限内不予答复', '对征收、征用决定及其补偿决定不服', '提起行政诉讼'))
      for p in ('不给答复', '办证窗口', '拖着不批', '拆迁补偿', '赔迁', '征地补偿')],
    *[(p, ('权利人明确同意', '住宅、宾馆房间等私密空间'))
      for p in ('闯进来', '私自进屋', '乱进我屋')],
    *[(p, ('任何组织或者个人不得以侮辱、诽谤等方式侵害他人的名誉权',))
      for p in ('群里说我', '全是编的', '污蔑我')],
    *[(p, ('第一顺序：配偶、子女、父母', '继承权男女平等'))
      for p in ('嫁出去', '外嫁女', '重男轻女')],
    # 2026-11-06 补的 7 个概念组：只认具体意图短语，不把泛词当成场景依据。
    *[(p, ('相邻', '通风', '采光', '日照', '妨碍'))
      for p in ('挡光', '遮光', '挡采光', '遮采光', '起高墙', '盖高墙', '建高墙', '高墙遮挡')],
    *[(p, ('从轻', '减轻', '行政处罚', '危害后果'))
      for p in ('少罚', '轻罚', '罚钱能少', '少罚点', '罚少点', '罚太多',
                '主动消除危害', '主动减轻危害', '从轻减轻', '减轻处罚', '从轻处罚')],
    *[(p, ('查封', '扣押', '无法使用', '解除合同'))
      for p in ('封住', '被封', '查封', '不让进', '门被封')],
    *[(p, ('撤销', '复核', '学位授予单位'))
      for p in ('撤销学位', '学位不算', '学位被撤', '收回学位', '撤销了学位', '不发学位')],
    *[(p, ('约定的方法', '租赁物', '损耗', '不承担赔偿责任'))
      for p in ('磨旧', '磨损', '用旧', '正常损耗')],
    *[(p, ('人身权益', '财产损失', '获得的利益', '赔偿数额'))
      for p in ('盗图', '盗发', '盗用照片', '盗发照片', '偷用头像', '乱发私照', '盗用我照片')],
    *[(p, ('法定职责', '不履行', '提起诉讼', '行政机关'))
      for p in ('不作为', '机关不理', '部门不理', '两三个月不理', '几个月不理',
                '拖着不理', '拖着不办', '迟迟不理', '不履行法定职责')],
])
def test_livelihood_short_aliases_keep_original_and_supply_legal_concepts(phrase, terms):
    """具体意图短语（如“撤销学位”“门被封”）可唤起场景概念并保留原话；裸名词不触发。"""
    query = phrase + '如何处理'
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + ' ')
    assert all(term in expanded for term in terms)


@pytest.mark.parametrize('phrase, terms', [
    *[(p, ('停工留薪期', '原工资福利待遇不变', '由所在单位按月支付'))
      for p in ('干活摔', '干活受伤', '工伤休息')],
    *[(p, ('应当自知道或者应当知道作出行政行为之日起六个月内提出', '直接向人民法院提起诉讼'))
      for p in ('接到处罚', '收到处罚', '不服处罚', '起诉期限')],
    *[(p, ('夫妻一方的个人财产', '一方的婚前财产'))
      for p in ('结婚前', '婚前买房', '婚前财产')],
    *[(p, ('立有数份遗嘱', '内容相抵触的', '以最后的遗嘱为准'))
      for p in ('两张遗嘱', '两份遗嘱', '多份遗嘱', '遗嘱不一样', '遗嘱冲突', '前后遗嘱', '几份遗嘱')],
    *[(p, ('网络服务提供者知道或者应当知道', '未采取必要措施', '承担连带责任'))
      for p in ('我的私照', '私照被发')],
    *[(p, ('第一次讯问', '有权委托辩护人', '只能委托律师作为辩护人'))
      for p in ('叫我去问话', '喊我去问话')],
    *[(p, ('当事人有权进行陈述和申辩', '不得因当事人陈述、申辩而给予更重的处罚'))
      for p in ('罚更多', '再顶嘴')],
])
def test_second_batch_livelihood_aliases_supply_legal_concepts(phrase, terms):
    """第二批通用概念：每个带动作边界的新入口都要补齐对应法定用语，不能只靠旧词偶然命中。"""
    query = phrase + '怎么办'
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + ' ')
    assert all(term in expanded for term in terms)


@pytest.mark.parametrize('query', [
    '骨折怎么康复', '一份遗嘱怎么写', '遗嘱怎么公证', '处罚是什么', '去法院旁听',
    '这个平台怎么用', '律师费怎么收', '私照怎么拍', '孩子顶嘴怎么办', '一张遗嘱有效吗',
])
def test_second_batch_bare_words_do_not_trigger_legal_scenes(query):
    """裸词“骨折/遗嘱/处罚/法院/平台/律师/私照/顶嘴”单独出现不足以认定场景；
    只有“干活摔、两张遗嘱、接到处罚、我的私照、再顶嘴”等带动作或冲突边界的短语才扩展。"""
    assert expand_colloquial_query(query) == query


@pytest.mark.parametrize('query', [
    '学位如何处理', '幼儿园学位怎么申请', '证件照如何处理',
    '罚钱怎么办', '退租怎么办', '照片如何处理', '赔多少合适',
    '不理怎么办', '去告还是算了', '告状有用吗', '高墙属于违建吗',
])
def test_generic_words_alone_do_not_trigger_the_seven_scenes(query):
    """泛词单独出现不足以认定场景：学位申请、证件照、普通罚款、正常退租、
    索赔、纠纷处理等问句保持原样，只有具体意图短语才扩展。"""
    assert expand_colloquial_query(query) == query


@pytest.mark.parametrize('query', ['电脑重装系统怎么办', '孩子顶嘴怎么办'])
def test_ambiguous_reinstall_and_talking_back_do_not_trigger_legal_scene(query):
    """裸词“重装”“顶嘴”多义，不把技术问题和普通家庭管教自动扩展成违约或行政处罚。"""
    assert expand_colloquial_query(query) == query


def test_combined_labor_query_expands_both_grounds():
    """多种口语同时出现时，欠薪与欠缴社保两个解除事由均不能遗漏。"""
    query = "公司拖欠工资不交社保解除合同"
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + " ")
    assert "未及时足额支付劳动报酬" in expanded
    assert "未依法为劳动者缴纳社会保险费" in expanded
    assert "解除劳动合同" in expanded


@pytest.mark.parametrize("query", ["", "今天天气怎么样", "未及时足额支付劳动报酬"])
def test_query_without_colloquial_alias_is_unchanged(query):
    """无别名的问题和已有法定表述保持原样，避免无关扩展污染 BM25。"""
    assert expand_colloquial_query(query) == query


@pytest.mark.parametrize("phrase", ["vpn", "VPN", "VPNcodex", "vPnCoDeX"])
def test_colloquial_expansion_matches_case_insensitively_and_keeps_original(phrase):
    """大小写或紧邻产品名不能阻断联网场景扩展；用户原问句的大小写保持原样。"""
    query = "我用的是" + phrase + "来做项目"
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + " ")
    assert "计算机信息网络直接进行国际联网" in expanded
    assert "国际出入口信道" in expanded and "自行建立或者使用其他信道" in expanded


def test_colloquial_expansion_deduplicates_terms_ignoring_case(monkeypatch):
    """别名和扩展短语同时按大小写不敏感比较，保留首次命中的原始拼写。"""
    monkeypatch.setattr("local_rag.retrieval.COLLOQUIAL_ALIASES", {
        "VPN": "International CHANNEL", "vpn": "international channel",
    })
    assert expand_colloquial_query("vPnCoDeX") == "vPnCoDeX International CHANNEL"


@pytest.mark.parametrize("aliases, query, expected", [
    ({"a": "Shared FIRST", "b": "shared SECOND"}, "AB", "AB Shared FIRST SECOND"),
    ({"礼": "返还 赠与", "彩礼": "返还 赠与 生活困难"}, "彩礼", "彩礼 返还 赠与 生活困难"),
])
def test_overlapping_alias_groups_keep_only_novel_phrases(monkeypatch, aliases, query, expected):
    """补充已存在的口语场景时，共同短语只出现一次，新要件仍追加且保留首次拼写。"""
    monkeypatch.setattr("local_rag.retrieval.COLLOQUIAL_ALIASES", aliases)
    assert expand_colloquial_query(query) == expected


@pytest.mark.parametrize("query, term", [
    ("还不给我交社保", "未依法为劳动者缴纳社会保险费"),
    ("还不给我缴社保", "未依法为劳动者缴纳社会保险费"),
    ("经济补偿按工作年限计算", "每满一年支付一个月工资的标准向劳动者支付"),
    ("配合验收但拒绝配合，是否违反协助义务", "按照约定全面履行自己的义务"),
    ("不履行合同，违约金比实际损失高，要求调低违约金", "请求人民法院或者仲裁机构予以适当减少"),
    ("约定还款日到期不还，一直不还", "借款人应当按照约定的期限返还借款"),
    ("还没有法院判决就说已经是罪犯，未经法院判决能定罪吗", "未经人民法院依法判决"),
    ("对象成了前任，分手后花了我不少钱和彩礼", "当事人请求返还按照习俗给付的彩礼"),
])
def test_scene_aliases_add_shared_statutory_terms_once(query, term):
    """夹字表达也能命中；同场景的多个同义触发词只追加一次相同术语。"""
    expanded = expand_colloquial_query(query)
    assert expanded.startswith(query + " ")
    assert expanded.count(term) == 1


@pytest.fixture(scope="module")
def corpus_retriever():
    """在本测试模块复用真实语料的内存词面索引，不构建向量或写入缓存。"""
    rows, _ = load_corpus()
    return Retriever(rows, Embedder())


@pytest.mark.parametrize("query, title, articles", [
    pytest.param(
        "签了供货合同后，对方说合同没写要配合验收，所以拒绝配合，我能要求他履行协助义务吗？",
        "中华人民共和国民法典", ("第五百零九条",), id="Q01"),
    pytest.param(
        "合同约定的违约金比实际损失高很多，对方又不履行合同，我能请求法院调低违约金吗？",
        "中华人民共和国民法典", ("第五百八十五条", "第五百七十七条"), id="Q05"),
    pytest.param(
        "借款已经到了约定还款日，对方一直不还，逾期利息的承担有什么依据？",
        "中华人民共和国民法典", ("第六百七十五条", "第六百七十六条"), id="Q08"),
    pytest.param(
        "公司拖欠工资又不交社保，我想解除劳动合同；如果符合补偿条件，经济补偿按工作年限怎么计算？",
        "中华人民共和国劳动合同法", ("第三十八条", "第四十七条"), id="Q16"),
    pytest.param(
        "公司拖欠工资，还不给我交社保，我想解除劳动合同，经济补偿按工作年限怎么计算？",
        "中华人民共和国劳动合同法", ("第三十八条", "第四十七条"), id="Q16-interleaved"),
    pytest.param(
        "家人被立案调查，还没有法院判决，办案人员就说他已经是罪犯，这种说法的法律边界是什么？",
        "中华人民共和国刑事诉讼法", ("第十二条",), id="Q36"),
])
def test_scene_core_articles_in_real_corpus_top_four(corpus_retriever, query, title, articles):
    """按服务实际 Top-4 验证全部目标条款；精确法名防止同号其他规范冒充命中。"""
    hits, mode = asyncio.run(corpus_retriever.search(query, top_k=4))
    actual = {(hit["title"], hit["locator"].split()[0]) for hit in hits}
    assert mode == "lexical" and len(hits) <= 4
    assert {(title, article) for article in articles} <= actual, (query, actual)


def test_relationship_property_query_recalls_criminal_and_civil_routes_in_top_four(corpus_retriever):
    """真实语料前四名须兼有刑事与民事线索；不将恋爱转账直接认定为诈骗或彩礼。"""
    query = "对象跟别人跑了，骗了我三万块，还花了我五万多怎么办"
    hits, mode = asyncio.run(corpus_retriever.search(query, top_k=4))
    actual = {(hit["title"], hit["locator"].split()[0]) for hit in hits}
    assert mode == "lexical" and len(hits) <= 4
    assert ("中华人民共和国刑法", "第二百六十六条") in actual, actual
    # 旧结果已有第148条，仍缺返还路径；额外要求第157条或彩礼返还条款，才能抓住本次偏科。
    civil_routes = {
        ("中华人民共和国民法典", "第一百五十七条"),
        ("最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）", "第五条"),
    }
    assert actual & civil_routes, actual


def test_composite_query_covers_four_legal_routes_with_general_dense_hits(corpus_retriever):
    """真实语料加泛化向量桩：四个目标均须靠口语扩展和融合召回，不预置到向量命中中。"""
    rows = corpus_retriever.rows
    retriever = Retriever(rows, Embedder())
    retriever.vectors = [[0.0, -1.0]] * len(rows)
    generic = [i for i, row in enumerate(rows)
               if row["title"] == "中华人民共和国网络安全法"
               and row["locator"].split()[0] in {"第四十四条", "第四十五条", "第四十六条", "第四十七条"}]
    assert len(generic) == 4
    for i in generic:
        retriever.vectors[i] = [0.0, 1.0]
    hits, mode = asyncio.run(retriever.search(COMPOSITE_QUERY, top_k=4))
    actual = {(hit["title"], hit["locator"].split()[0]) for hit in hits}
    expected = {
        ("中华人民共和国计算机信息网络国际联网管理暂行规定", "第六条"),
        ("计算机软件保护条例", "第二十三条"),
        ("中华人民共和国刑法", "第二百一十九条"),
        ("中华人民共和国反不正当竞争法", "第十条"),
    }
    assert mode == "hybrid" and len(hits) == 4
    assert expected <= actual, actual


def test_relationship_hybrid_keeps_criminal_route_with_many_civil_dense_hits(corpus_retriever):
    """向量路被同一部民法典占满时，仍保留刑事核心条款与民事路径，不单法挤满前四。"""
    rows = corpus_retriever.rows
    retriever = Retriever(rows, Embedder())
    retriever.vectors = [[0.0, -1.0]] * len(rows)
    civil = [i for i, row in enumerate(rows) if row["title"] == "中华人民共和国民法典"
             and row["locator"].split()[0] in {"第一百四十八条", "第一百四十九条", "第一百五十条", "第一百五十七条"}]
    assert len(civil) == 4
    for i in civil:
        retriever.vectors[i] = [0.0, 1.0]
    hits, mode = asyncio.run(retriever.search("对象跟别人跑了，骗了我三万块，还花了我五万多怎么办", top_k=4))
    actual = {(hit["title"], hit["locator"].split()[0]) for hit in hits}
    assert mode == "hybrid"
    assert ("中华人民共和国刑法", "第二百六十六条") in actual, actual
    assert actual & {("中华人民共和国民法典", "第一百四十八条"),
                     ("中华人民共和国民法典", "第一百五十七条")}, actual
    assert sum(hit["title"] == "中华人民共和国民法典" for hit in hits) <= 2


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
@pytest.mark.parametrize("query, required, alternatives, dense_title, dense_articles", [
    pytest.param(
        "买二手车结果发现是泡水车，车商当时拍胸脯说没出过事故，我能退一赔三吗",
        {("中华人民共和国消费者权益保护法", "第五十五条"), ("中华人民共和国民法典", "第一百四十八条")}, set(),
        "中华人民共和国道路交通安全法", {"第七十六条", "第七十七条", "第七十八条", "第七十九条"}, id="used-car-fraud"),
    pytest.param(
        "邻居把我家养的宠物狗投毒毒死了，报警警察说不管让我去起诉，怎么维权",
        {("中华人民共和国刑法", "第二百七十五条"), ("中华人民共和国民法典", "第二百三十八条")}, set(),
        "中华人民共和国民法典", {"第一千二百四十五条", "第一千二百四十六条", "第一千二百四十七条", "第一千二百四十八条"}, id="poisoned-pet"),
    pytest.param(
        "网上有人把我的照片和身份证号发到群里造谣骂我，怎么让他删掉并赔偿",
        set(), {("中华人民共和国民法典", "第一千零二十四条"), ("中华人民共和国民法典", "第一千一百九十四条"),
                ("中华人民共和国民法典", "第一千一百九十五条"), ("中华人民共和国刑法", "第二百四十六条")},
        "中华人民共和国居民身份证法", {"第一条", "第二条", "第三条", "第四条"}, id="online-defamation"),
])
def test_everyday_cases_cover_core_routes_in_real_corpus_top_four(
        corpus_retriever, monkeypatch, mode, query, required, alternatives, dense_title, dense_articles):
    """真实语料验证民生跨法域召回；混合模式仅给偏题条款配置向量，不预置任何目标命中。"""
    if mode == "hybrid":
        rows = corpus_retriever.rows
        biased = [i for i, row in enumerate(rows) if row["title"] == dense_title
                  and row["locator"].split()[0] in dense_articles]
        assert len(biased) == 4, (dense_title, biased)
        vectors = [[0.0, -1.0]] * len(rows)
        for i in biased:
            vectors[i] = [0.0, 1.0]
        # 自动恢复共享检索器，避免模拟向量影响其他真实语料测试。
        monkeypatch.setattr(corpus_retriever, "vectors", vectors)
    hits, actual_mode = asyncio.run(corpus_retriever.search(query, top_k=4))
    actual = {(hit["title"], hit["locator"].split()[0]) for hit in hits}
    assert actual_mode == mode and len(hits) == 4
    assert required <= actual, (query, actual_mode, actual)
    if alternatives:
        assert alternatives & actual, (query, actual_mode, actual)


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_school_certificate_dispute_recalls_student_rights_in_top_four(corpus_retriever, monkeypatch, mode):
    """真实语料验证领证和学生权利；合成向量仅给收回/无效证书路径，不预填目标条款。"""
    if mode == "hybrid":
        rows = corpus_retriever.rows
        biased_targets = {("中华人民共和国教育法", "第八十二条"), ("中华人民共和国教育法", "第八十三条"),
                          ("中华人民共和国学位法", "第三十七条"), ("中华人民共和国学位法", "第三十八条")}
        biased = [i for i, row in enumerate(rows) if (row["title"], row["locator"].split()[0]) in biased_targets]
        assert len(biased) == 4
        vectors = [[0.0, -1.0]] * len(rows)
        for i in biased:
            vectors[i] = [0.0, 1.0]
        monkeypatch.setattr(corpus_retriever, "vectors", vectors)
    query = "学校的老师不让我拿毕业证这个应该怎么办的呢我不就是没有答应那时候和他们参加比赛不是的吗"
    hits, actual_mode = asyncio.run(corpus_retriever.search(query, top_k=4))
    actual = {(hit["title"], hit["locator"].split()[0]) for hit in hits}
    assert actual_mode == mode and len(hits) == 4
    assert ("中华人民共和国教育法", "第四十三条") in actual, actual
    assert actual & {("中华人民共和国高等教育法", "第五十八条"),
                     ("中华人民共和国高等教育法", "第二十条")}, actual


@pytest.mark.parametrize("query, expected", [
    ("我在酒店客房的插座里发现针孔摄像头，怎么维权", {("公共安全视频图像信息系统管理条例", "第八条")}),
    ("健身房充卡后老板跑路了，我充值的钱还能退吗", {("中华人民共和国消费者权益保护法", "第五十三条"),
                                            ("中华人民共和国消费者权益保护法实施条例", "第二十二条")}),
    ("我是城市户口，父母留下的农村老房子拆迁，村里说城里人不给补偿，怎么办", {("中华人民共和国土地管理法实施条例", "第三十六条"),
                                                            ("中华人民共和国土地管理法", "第四十八条")}),
])
def test_general_remedies_outrank_unmentioned_special_laws_in_real_corpus(corpus_retriever, monkeypatch, query, expected):
    """真实条文加有序合成向量：目标位于专门法之后，验证候选重排而非真实向量召回能力。"""
    rows = corpus_retriever.rows
    special_titles = ("中华人民共和国未成年人保护法", "中华人民共和国工会法", "娱乐场所管理条例", "中华人民共和国归侨侨眷权益保护法实施办法",
                      "长江三峡工程建设移民条例", "大中型水利水电工程建设征地补偿和移民安置条例", "蓄滞洪区运用补偿办法")
    noisy = []
    for title in special_titles:
        candidates = [i for i, row in enumerate(rows) if row["title"] == title]
        assert candidates, title
        noisy.append(next((i for i in candidates if any(term in rows[i]["text"] for term in ("有权请求", "赔偿损失", "应当予以返还"))), candidates[0]))
    targets = [i for i, row in enumerate(rows) if (row["title"], row["locator"].split()[0]) in expected]
    assert len(targets) == len(expected)
    vectors = [[0.0, -1.0]] * len(rows)
    for rank, i in enumerate(noisy + targets):
        similarity = 1.0 - rank * 0.02
        vectors[i] = [math.sqrt(1.0 - similarity ** 2), similarity]
    monkeypatch.setattr(corpus_retriever, "vectors", vectors)
    hits, mode = asyncio.run(corpus_retriever.search(query, top_k=4))
    actual = {(hit["title"], hit["locator"].split()[0]) for hit in hits}
    assert mode == "hybrid" and expected <= actual, (query, actual)
    assert hits[0]["title"] not in set(special_titles[:2] + special_titles[3:4])


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
@pytest.mark.parametrize("query, required, alternatives, biased", [
    pytest.param(
        "报了个云南低价旅游团，导游逼我们买翡翠，我不买就把我赶下大巴车扔在半路怎么办",
        set(), {("中华人民共和国旅游法", "第三十五条"), ("导游人员管理条例", "第十六条")},
        {("中国公民出国旅游管理办法", f"第{number}条") for number in ("十三", "十四", "十五", "十六")}, id="domestic-tour"),
    pytest.param(
        "我家十岁小孩偷拿外婆手机给直播间主播刷了三万块钱礼物，平台说已经消费了不给退怎么办",
        {("未成年人网络保护条例", "第四十四条")},
        {("中华人民共和国民法典", article) for article in ("第十九条", "第一百四十五条", "第一百五十七条")},
        {("未成年人网络保护条例", article) for article in ("第四十一条", "第四十二条", "第四十三条", "第四十四条")}, id="minor-live-gifts"),
    pytest.param(
        "我刚买了一套二手房交完定金和首付，邻居告诉我这房子去年死过人，中介和房东当时都没说，我能退房吗",
        set(), {("中华人民共和国民法典", article) for article in ("第一百四十八条", "第五百条", "第九百六十二条")},
        {("中华人民共和国民法典", "第七百三十二条")}, id="concealed-house-facts"),
    pytest.param(
        "我家装修找的装修公司签了全包合同，拆完墙他们突然说要加两万块钱才继续干，不给就停工怎么办",
        set(), {("中华人民共和国民法典", article) for article in ("第五百四十三条", "第五百七十七条")},
        {("建设工程质量管理条例", "第六十九条")}, id="renovation-price-change"),
])
def test_second_round_scenarios_recall_general_rights_in_real_corpus(
        corpus_retriever, monkeypatch, mode, query, required, alternatives, biased):
    """真实BM25与偏题合成向量分别验证Top-4；民法典目标未预填向量，不代替线上召回评测。"""
    if mode == "hybrid":
        rows = corpus_retriever.rows
        indices = [i for i, row in enumerate(rows) if (row["title"], row["locator"].split()[0]) in biased]
        assert len(indices) == len(biased), biased
        vectors = [[0.0, -1.0]] * len(rows)
        for i in indices:
            vectors[i] = [0.0, 1.0]
        monkeypatch.setattr(corpus_retriever, "vectors", vectors)
    hits, actual_mode = asyncio.run(corpus_retriever.search(query, top_k=4))
    actual = {(hit["title"], hit["locator"].split()[0]) for hit in hits}
    assert actual_mode == mode and len(hits) == 4
    assert required <= actual and alternatives & actual, actual
    if "云南" in query:
        assert all(hit["title"] != "中国公民出国旅游管理办法" for hit in hits)


def test_ordinary_company_dismissal_does_not_promote_state_owned_surplus_rules(corpus_retriever, monkeypatch):
    """真实两类候选均入向量池时，普通辞退问题中劳动合同法须压过排在前面的国企专法。"""
    rows = corpus_retriever.rows
    indices = [i for i, row in enumerate(rows) if row["title"] == "国有企业富余职工安置规定"][:4]
    assert len(indices) == 4
    general = [i for i, row in enumerate(rows) if row["title"] == "中华人民共和国劳动合同法"
               and row["locator"].split()[0] in ("第三十九条", "第四十六条", "第四十七条", "第四十八条")]
    assert len(general) == 4
    vectors = [[0.0, -1.0]] * len(rows)
    for rank, i in enumerate(indices + general):
        similarity = 1.0 - rank * 0.02
        vectors[i] = [math.sqrt(1.0 - similarity ** 2), similarity]
    monkeypatch.setattr(corpus_retriever, "vectors", vectors)
    hits, mode = asyncio.run(corpus_retriever.search("普通公司突然辞退我，说年终奖一分不给怎么办", top_k=4))
    assert mode == "hybrid" and any(hit["title"] == "中华人民共和国劳动合同法" for hit in hits)
    assert all(hit["title"] != "国有企业富余职工安置规定" for hit in hits)


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_street_assault_and_counter_accusation_recall_both_paths(corpus_retriever, monkeypatch, mode):
    """真实语料Top-4须兼顾受侵害与被指控两条路径；合成向量不预填目标法条。"""
    if mode == "hybrid":
        rows = corpus_retriever.rows
        # 按实测观察的相对次序排列六条偏题候选；目标237/243/50均不预填到向量池。
        biased = [("中华人民共和国刑法", "第二百三十六条"), ("中华人民共和国民法典", "第一千零一十条"),
                  ("中华人民共和国妇女权益保障法", "第二十三条"), ("中华人民共和国妇女权益保障法", "第二十九条"),
                  ("中华人民共和国反家庭暴力法", "第三十三条"), ("中华人民共和国反家庭暴力法", "第二十条")]
        by_article = {(row["title"], row["locator"].split()[0]): i for i, row in enumerate(rows)}
        assert all(article in by_article for article in biased)
        vectors = [[0.0, -1.0]] * len(rows)
        for rank, article in enumerate(biased):
            similarity = 1.0 - rank * 0.02
            vectors[by_article[article]] = [math.sqrt(1.0 - similarity ** 2), similarity]
        monkeypatch.setattr(corpus_retriever, "vectors", vectors)
    query = "我在回家的路上碰到了几个女的她们几个女的一起把我给那个了，时候她们告我强奸她们怎么办，这个事情应该怎么解决的呢"
    hits, actual_mode = asyncio.run(corpus_retriever.search(query, top_k=4))
    actual = {(hit["title"], hit["locator"].split()[0]) for hit in hits}
    assert actual_mode == mode and len(hits) == 4
    assert {("中华人民共和国刑法", "第二百三十七条"), ("中华人民共和国刑法", "第二百四十三条"),
            ("中华人民共和国治安管理处罚法", "第五十条")} <= actual, actual
    assert not actual & {("中华人民共和国民法典", "第一千零一十条"),
                         ("中华人民共和国妇女权益保障法", "第二十三条")}, actual


def test_compound_independent_anchors_survive_single_semantic_consensus(monkeypatch):
    """两组独立法定要件的强分支须共同保留；不按具体法名或条号写死优先级。"""
    monkeypatch.setattr('local_rag.retrieval.COLLOQUIAL_ALIASES', {'受害口语': '第一分支规范', '反咬口语': '第二分支规范'})
    rows = [dict(id=str(i), title='同部规范', text='中性文本') for i in range(5)]
    retriever = Retriever(rows, Embedder())
    retriever.vectors = [[0., 1.]]
    class Store:
        async def search(self, vector, limit):
            return [(0, .9), (1, .85), (2, .8)]
    retriever.store = Store()
    def rank(query):
        if query == '第一分支规范': return [(3, 120.), (0, 50.)]
        if query == '第二分支规范': return [(4, 110.), (1, 40.)]
        return [(0, 170.), (1, 160.), (2, 150.), (3, 140.), (4, 130.)]
    monkeypatch.setattr(retriever._bm25, 'rank', rank)
    hits, _ = asyncio.run(retriever.search('受害口语之后反咬口语'))
    assert {'3', '4'} <= {h['id'] for h in hits}


def test_compound_weak_prefix_does_not_consume_valid_intent_slots(monkeypatch):
    aliases = {f'弱{i}': f'弱规范{i}' for i in range(4)}
    aliases.update({'受害': '人身安全禁止侵害', '反咬': '捏造事实调查证据'})
    monkeypatch.setattr('local_rag.retrieval.COLLOQUIAL_ALIASES', aliases)
    retriever = Retriever([dict(id=str(i), title='规范', text='文本') for i in range(5)], Embedder())
    monkeypatch.setattr(retriever._bm25, 'rank', lambda q: [(3, 120.), (0, 30.)] if q=='人身安全禁止侵害'
                        else [(4, 110.), (0, 30.)] if q=='捏造事实调查证据' else [(0, 20.)])
    assert retriever._compound_intent_anchors('弱0弱1弱2弱3受害后反咬', {3: 150., 4: 140.}, lambda i: 1.) == [3, 4]


@pytest.mark.parametrize("title, supported_query", [
    ("中华人民共和国未成年人保护法", "未成年人如何维护权益"),
    ("中华人民共和国预防未成年人犯罪法", "孩子如何维护权益"),
    ("中华人民共和国工会法", "工会如何维护权益"),
    ("娱乐场所管理条例", "KTV消费纠纷如何处理"),
    ("归侨侨眷权益保护法实施办法", "归侨如何维护权益"),
    ("长江三峡工程建设移民条例", "三峡移民安置如何处理"),
    ("大中型水利水电工程建设征地补偿和移民安置条例", "水库征地补偿如何处理"),
    ("蓄滞洪区运用补偿办法", "蓄滞洪区补偿如何处理"),
    ("中国公民出国旅游管理办法", "出国旅游纠纷如何处理"),
    ("中国公民出国旅游管理办法", "境外旅游纠纷如何处理"),
    ("国有企业富余职工安置规定", "国企富余职工安置如何处理"),
])
@pytest.mark.parametrize("supported", [False, True])
def test_special_scope_controls_dense_and_strong_lexical_weights(monkeypatch, title, supported_query, supported):
    """未提及特定主体/工程时两路均降权；明确提及后恢复救济与超强词面的完整贡献。"""
    rows = [{"id": "special", "title": title, "text": "当事人有权请求救济。"},
            {"id": "general", "title": "一般规范", "text": "一般条款。"}]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(0, 100.0)])
    retriever.vectors = [[0.0, 1.0]] * 2
    query = supported_query if supported else "酒店纠纷如何维护权益"
    hits, mode = asyncio.run(retriever.search(query, top_k=2))
    assert mode == "hybrid"
    assert hits[0]["id"] == ("special" if supported else "general")
    by_id = {hit["id"]: hit["retrieval_score"] for hit in hits}
    expected = (1.2 * 1.35 + 1.35) / 31 if supported else (1.2 * 1.35 * 0.4 + 0.25 * 0.4) / 31
    assert by_id["special"] == pytest.approx(round(expected, 5))


def test_restricted_special_titles_do_not_trigger_diversity_or_change_lexical_fallback(monkeypatch):
    """越界高分不能计入超强锚点法名数；范围门控仅在有向量的混合模式中生效。"""
    rows = [{"id": "special-a", "title": "中华人民共和国工会法", "text": "当事人有权请求救济。"},
            {"id": "special-b", "title": "归侨侨眷权益保护法实施办法", "text": "当事人有权请求救济。"}] + [
        {"id": f"general-{i}", "title": "一般规范", "text": "一般条款。"} for i in range(4)]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(0, 100.0), (1, 99.0)])
    hits, mode = asyncio.run(retriever.search("酒店纠纷如何维护权益", top_k=2))
    assert mode == "lexical" and [(hit["id"], hit["retrieval_score"]) for hit in hits] == [("special-a", 100.0), ("special-b", 99.0)]
    retriever.vectors = [[0.0, 1.0]] * len(rows)
    hits, mode = asyncio.run(retriever.search("酒店纠纷如何维护权益", top_k=4))
    assert mode == "hybrid" and [hit["id"] for hit in hits] == [f"general-{i}" for i in range(4)]


def test_bm25_rank_keeps_expansion_at_search_boundary():
    """BM25 独立入口仍只匹配传入文本；显式扩展后才召回社保法定表述。"""
    rows = [{"id": "social-insurance", "text": "未依法为劳动者缴纳社会保险费"}]
    assert bm25_rank(rows, "不交社保") == []
    ranked = bm25_rank(rows, expand_colloquial_query("不交社保"))
    assert ranked[0][0] == 0 and ranked[0][1] > 0


def test_search_recalls_labor_contract_article_38_from_corpus():
    """用本地核验语料验证核心条款进入前 12 候选；不调用向量服务或写入索引。"""
    async def run():
        rows, _ = load_corpus()
        retriever = Retriever(rows, Embedder())
        for query in ("拖欠工资", "不交社保", "拖欠工资不交社保解除合同"):
            hits, mode = await retriever.search(query, top_k=12)
            assert mode == "lexical"
            assert any(
                hit["title"] == "中华人民共和国劳动合同法"
                and hit["locator"].split()[0] == "第三十八条"
                for hit in hits
            ), (query, [(hit["title"], hit["locator"]) for hit in hits])

    asyncio.run(run())


def test_hybrid_search_embeds_original_colloquial_query():
    """扩展只影响词面路；向量路继续使用原问句，同时保留 Top-K 边界。"""
    async def run():
        rows = [{"id": "social-insurance", "text": "未依法为劳动者缴纳社会保险费"}]
        embedder = Embedder()
        retriever = Retriever(rows, embedder)
        # 直接提供内存向量以隔离网络与磁盘副作用；embed() 仍记录真实调用参数。
        retriever.vectors = [[0.0, 1.0]]
        query = "不交社保"
        hits, mode = await retriever.search(query, top_k=1)
        assert mode == "hybrid" and [hit["id"] for hit in hits] == ["social-insurance"]
        assert embedder.calls == [([query], True)]
        assert await retriever.search(query, top_k=0) == ([], "lexical")
        assert await retriever.search(query, top_k=-1) == ([], "lexical")
        assert embedder.calls == [([query], True)]

    asyncio.run(run())


def test_rrf_keeps_both_sources():
    """测试 RRF 两路融合算法：综合关键词名次与向量名次，避免单一算法绝对分数的偏差"""
    # 词面排名 [0, 1]，向量排名 [1, 2]，综合后条目 1 在两路均靠前，应当排第一
    assert fuse_ranks([0, 1], [1, 2]) == [
        (1, 1 / 32 + 1 / 31), (0, 1 / 31), (2, 1 / 32),
    ]


@pytest.mark.parametrize("use_store", [False, True], ids=["hybrid", "hybrid_qdrant"])
def test_intent_reranking_recovers_late_remedies_from_twenty_candidates(monkeypatch, use_store):
    """无别名、无词面命中时，第14/19名的权利条款也须进入池并超过错位监管罚则。"""
    query = "我该如何维护自身合法权益"
    assert expand_colloquial_query(query) == query
    rows = [{"id": str(i), "title": f"候选规范{i}", "text": "宣布证书无效，责令收回。"} for i in range(21)]
    rows[13]["text"] = "受教育者享有下列权利，可以提出申诉。"
    rows[18]["text"] = "对相关决定不服，可以依法提起诉讼。"

    class Store:
        requested = None

        async def search(self, vector, limit):
            self.requested = limit
            return [(i, 1.0) for i in range(21)][:limit]

    store = Store() if use_store else None
    retriever = Retriever(rows, Embedder(), store=store)
    monkeypatch.setattr(retriever._bm25, "rank", lambda text: [])
    retriever.vectors = [[0.0, 1.0]] * len(rows)
    hits, mode = asyncio.run(retriever.search(query, top_k=21))
    assert mode == ("hybrid_qdrant" if use_store else "hybrid")
    assert len(hits) == 20 and "20" not in {hit["id"] for hit in hits}
    assert [hit["id"] for hit in hits[:2]] == ["13", "18"]
    assert hits[0]["retrieval_score"] == pytest.approx(round(1.2 * 1.35 / 44, 5))
    assert hits[1]["retrieval_score"] == pytest.approx(round(1.2 * 1.35 / 49, 5))
    if use_store:
        assert store.requested == 20


@pytest.mark.parametrize("marker", [
    "享有下列权利", "有权请求", "提出申诉", "依法提起诉讼", "承担民事责任", "赔偿损失", "停止侵害",
    "有权查阅", "有权复印", "不得采取停止供电", "准予毕业", "应当予以返还", "请求人民法院", "申请复核",
    "履行更换、修理", "应当受理",
])
def test_remedy_structure_boosts_dense_candidate_without_aliases(monkeypatch, marker):
    """每类救济结构须影响真实搜索输出，不能仅校验常量中是否抄入对应文本。"""
    rows = [{"id": "neutral", "title": "规范甲", "text": "一般说明。"},
            {"id": "remedy", "title": "规范乙", "text": "规定：" + marker + "。"}]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [])
    retriever.vectors = [[0.0, 1.0]] * 2
    hits, mode = asyncio.run(retriever.search("我该如何维护自身合法权益", top_k=2))
    assert mode == "hybrid" and [hit["id"] for hit in hits] == ["remedy", "neutral"]
    assert hits[0]["retrieval_score"] == pytest.approx(round(1.2 * 1.35 / 32, 5))


@pytest.mark.parametrize("reference", ["本法规定", "本条例规定", "本规定", "本办法"])
def test_generic_liability_without_lexical_support_does_not_displace_dense_top4(monkeypatch, reference):
    """泛责任模板不能凭结构加分挤掉语义前几名；任意规范标题都适用。"""
    rows = [dict(id=f"specific-{i}", title=f"规范{i}", text="具体行为及处理条件。") for i in range(4)]
    rows += [dict(id=f"generic-{i}", title=f"其他规范{i}",
                  text=f"违反{reference}，给他人造成损害的，依法承担民事责任；构成犯罪的，依法追究刑事责任。")
             for i in range(4)]
    retriever = Retriever(rows, Embedder())
    retriever.vectors = [[0., 1.]]
    class Store:
        async def search(self, vector, limit):
            return [(i, .8 - i * .01) for i in range(8)]
    retriever.store = Store()
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [])
    hits, mode = asyncio.run(retriever.search("遇到这种事该如何处理", top_k=8))
    assert mode == 'hybrid_qdrant'
    assert [h['id'] for h in hits[:4]] == [f'specific-{i}' for i in range(4)]
    assert hits[4]['retrieval_score'] == round(1.2 / 35, 5)
    assert len(hits) == 8  # 只取消无依据的加分，不删除候选。


@pytest.mark.parametrize("score, factor", [(34.999, 1.0), (35.0, 1.35)])
def test_generic_liability_uses_existing_lexical_evidence_boundary(monkeypatch, score, factor):
    """沿用既有35分词面支持界限，有支持的责任条款仍可按原规则提权。"""
    rows = [dict(id='rule', title='匿名规范', text='违反本法规定，造成损害的，依法承担民事责任。')]
    retriever = Retriever(rows, Embedder())
    retriever.vectors = [[0., 1.]]
    monkeypatch.setattr(retriever._bm25, 'rank', lambda query: [(0, score)])
    hits, _ = asyncio.run(retriever.search('遇到这种事该如何处理'))
    assert hits[0]['retrieval_score'] == round((1.2 * factor + 1.0) / 31, 5)


@pytest.mark.parametrize("marker", [
    "宣布证书无效", "弄虚作假或者剽窃他人教学成果", "有关部门的工作人员违反本法规定，滥用职权",
    "在教师资格认定工作中玩忽职守", "监督管理职责的部门及其工作人员，必须对在履行职责中知悉",
    "有关主管部门及其工作人员对在履行职责中知悉",
])
def test_mismatched_official_structure_is_downweighted_for_personal_remedy(monkeypatch, marker):
    """普通维权问句不应被排在向量首位的错位官方内部罚则压住。"""
    rows = [{"id": "official", "title": "规范甲", "text": marker + "。"},
            {"id": "neutral", "title": "规范乙", "text": "一般说明。"}]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [])
    retriever.vectors = [[0.0, 1.0]] * 2
    hits, mode = asyncio.run(retriever.search("我该如何维护自身合法权益", top_k=2))
    assert mode == "hybrid" and [hit["id"] for hit in hits] == ["neutral", "official"]
    assert hits[1]["retrieval_score"] == pytest.approx(round(1.2 * 0.55 / 31, 5))


@pytest.mark.parametrize("keyword", ["部门", "机关", "工作人员", "履职", "履行职责", "执法", "调查", "政务", "伪造", "假证", "弄虚作假"])
def test_official_query_disables_both_remedy_boost_and_penalty(monkeypatch, keyword):
    """用户明确查询官方/假证责任时，两类结构均不调节，保留原向量次序和贡献。"""
    rows = [{"id": "official", "title": "规范甲", "text": "宣布证书无效，责令收回。"},
            {"id": "remedy", "title": "规范乙", "text": "当事人有权请求救济。"}]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [])
    retriever.vectors = [[0.0, 1.0]] * 2
    hits, mode = asyncio.run(retriever.search("请说明" + keyword + "相关责任", top_k=2))
    assert mode == "hybrid" and [hit["id"] for hit in hits] == ["official", "remedy"]
    assert [hit["retrieval_score"] for hit in hits] == [round(1.2 / 31, 5), round(1.2 / 32, 5)]


def test_mismatched_structure_takes_priority_over_remedy_in_same_article(monkeypatch):
    """同条同时含两类特征时先压降错位监管，不因出现“有权请求”又恢复提权。"""
    rows = [{"id": "mixed", "title": "规范甲", "text": "宣布证书无效；当事人有权请求复核。"},
            {"id": "remedy", "title": "规范乙", "text": "当事人有权请求救济。"},
            {"id": "neutral", "title": "规范丙", "text": "一般说明。"}]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [])
    retriever.vectors = [[0.0, 1.0]] * 3
    hits, mode = asyncio.run(retriever.search("我该如何维护自身合法权益", top_k=3))
    assert mode == "hybrid" and [hit["id"] for hit in hits] == ["remedy", "neutral", "mixed"]
    assert hits[-1]["retrieval_score"] == pytest.approx(round(1.2 * 0.55 / 31, 5))


@pytest.mark.parametrize("titles, noise_first", [
    (("领域甲规范", "领域乙规范"), True),
    (("任意改名后的规范", "另一部匿名规范"), False),
    (("无业务词的标题A", "无业务词的标题B"), True),
])
def test_search_weighting_preserves_original_fusion_contract_across_names(monkeypatch, titles, noise_first):
    """原两参数融合函数可原样保留；混合加权不依赖法规名称或低行号噪音的排序优势。"""
    def original_two_argument_fusion(lexical, dense):
        # 模拟原接口，拒绝扩展参数；搜索端必须独立处理语义权重。
        return fuse_ranks(lexical, dense)

    monkeypatch.setattr("local_rag.retrieval.fuse_ranks", original_two_argument_fusion)
    rows = [
        {"id": "noise", "title": titles[0], "text": "偶发片段"},
        {"id": "semantic", "title": titles[1], "text": "语义条款"},
    ]
    if not noise_first:
        rows.reverse()
    noise = next(i for i, row in enumerate(rows) if row["id"] == "noise")
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(noise, 20.0)])
    retriever.vectors = [[0.0, -1.0] if row["id"] == "noise" else [0.0, 1.0] for row in rows]
    hits, mode = asyncio.run(retriever.search("请说明如何处理", top_k=2))
    assert mode == "hybrid"
    assert [hit["id"] for hit in hits] == ["semantic", "noise"]
    assert [hit["retrieval_score"] for hit in hits] == [round(1.2 / 31, 5), round(0.25 / 31, 5)]


@pytest.mark.parametrize("use_store", [False, True], ids=["hybrid", "hybrid_qdrant"])
def test_hybrid_semantics_outrank_isolated_low_score_daily_words(use_store):
    """使用真实 BM25 构造偶发碎片命中；两种向量后端都应优先保留语义与双路目标。"""
    rows = [
        {"id": "fund", "title": "国家自然科学基金条例", "text": "原创"},
        {"id": "gold", "title": "金银管理条例", "text": "别人"},
        {"id": "port", "title": "国境口岸卫生监督办法", "text": "卫生间"},
        {"id": "copyright", "title": "信息网络传播权保护条例", "text": "权利人享有信息网络传播权，未经许可不得向公众提供作品。"},
        {"id": "rental", "title": "住房租赁条例", "text": "住房租赁的出租人应当提供合法住房并履行合同。"},
    ]
    query = "别人把我的原创作品放到网上，租的房子卫生间还出了问题怎么办"
    lexical = bm25_rank(rows, query)
    scores = dict(lexical)
    assert all(0 < scores[i] < 35.0 for i in (0, 1, 2))
    assert {i for i, _ in lexical[:3]} == {0, 1, 2}
    assert 3 in scores and 4 not in scores

    class Store:
        async def search(self, vector, limit):
            return [(4, 1.0), (3, 0.8), (0, 0.2)]

    retriever = Retriever(rows, Embedder(), store=Store() if use_store else None)
    # 合成单位向量隔离外部服务；词面索引和融合均使用生产实现，不构建磁盘缓存。
    retriever.vectors = [[0.0, -1.0]] * 3 + [[0.6, 0.8], [0.0, 1.0]]
    hits, mode = asyncio.run(retriever.search(query, top_k=5))
    assert mode == ("hybrid_qdrant" if use_store else "hybrid")
    assert [hit["id"] for hit in hits[:2]] == ["copyright", "rental"]
    by_id = {hit["id"]: hit["retrieval_score"] for hit in hits}
    assert by_id["rental"] == pytest.approx(round(1.2 / 31, 5))
    assert by_id["copyright"] == pytest.approx(round(1 / 34 + 1.2 / 32, 5))
    assert by_id["fund"] == pytest.approx(round(0.25 / 32, 5))


@pytest.mark.parametrize("score, same_law, expected", [
    (34.999, False, 0.25 / 31), (35.0, False, 1.0 / 31), (19.0, True, 1.0 / 31),
])
def test_hybrid_strong_lexical_or_same_law_match_keeps_full_weight(monkeypatch, score, same_law, expected):
    """精确验证 35 分边界，以及低分但获同法印证的核心条款仍保留完整贡献并进入前两名。"""
    rows = [
        {"id": "core", "title": "合同核心规范", "text": "核心词项"},
        {"id": "noise", "title": "无关规范", "text": "日常碎片"},
        {"id": "semantic", "title": "租赁规范", "text": "租赁条款"},
    ]
    if same_law:
        rows[2]["title"] = rows[0]["title"]
    retriever = Retriever(rows, Embedder())
    # 只控制 BM25 边界分值，向量过滤、融合及结果包装仍走真实 search()。
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(0, score), (1, 10.0)])
    retriever.vectors = [[0.0, -1.0], [0.0, -1.0], [0.0, 1.0]]
    hits, mode = asyncio.run(retriever.search("咨询处理方式", top_k=3))
    assert mode == "hybrid" and [hit["id"] for hit in hits] == ["semantic", "core", "noise"]
    assert hits[1]["retrieval_score"] == pytest.approx(round(expected, 5))


@pytest.mark.parametrize("title_rank, expected_weight", [(0, 1.0), (5, 1.0), (6, 0.25)])
def test_hybrid_same_law_support_is_limited_to_dense_top_six(monkeypatch, title_rank, expected_weight):
    """同法核心条款可获完整词面权重；第六名提供印证，第七名不能扩大同法放行范围。"""
    rows = [
        {"id": "core", "title": "核心规范", "text": "核心条款"},
        {"id": "noise", "title": "无关规范", "text": "日常碎片"},
    ] + [{"id": f"dense-{i}", "title": f"语义规范{i}", "text": "语义条款"} for i in range(7)]
    rows[2 + title_rank]["title"] = "核心规范"
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(1, 20.0), (0, 19.0)])
    retriever.vectors = [[0.0, -1.0]] * 2 + [[0.6, 0.8]] * 7
    hits, mode = asyncio.run(retriever.search("咨询处理方式", top_k=9))
    by_id = {hit["id"]: hit["retrieval_score"] for hit in hits}
    assert mode == "hybrid"
    assert by_id["core"] == pytest.approx(round(expected_weight / 32, 5))
    if title_rank == 0:
        # 获同法印证的词面核心条款应排在无关碎片之前。
        assert [hit["id"] for hit in hits].index("core") < [hit["id"] for hit in hits].index("noise")


def test_hybrid_exact_consensus_beyond_top_six_does_not_need_title(monkeypatch):
    """精确双路共识不受前六法名限制；缺失法名不能让其他无标题碎片获得同法权重。"""
    rows = [{"id": str(i), "text": "合成条款"} for i in range(8)]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(1, 20.0), (0, 19.0)])
    retriever.vectors = [[0.8, 0.6], [0.0, -1.0]] + [[0.6, 0.8]] * 6
    hits, mode = asyncio.run(retriever.search("咨询处理方式", top_k=8))
    assert mode == "hybrid" and hits[0]["id"] == "0"
    by_id = {hit["id"]: hit["retrieval_score"] for hit in hits}
    assert by_id["0"] == pytest.approx(round(1 / 32 + 1.2 / 37, 5))
    assert by_id["1"] == pytest.approx(round(0.25 / 31, 5))


@pytest.mark.parametrize("score, title, expected_weight, first", [
    (84.999, "核心规范", 1.0, "semantic"),
    (85.0, "核心规范", 1.35, "core"),
    (100.0, "", 1.0, "semantic"),
    (100.0, None, 1.0, "semantic"),
])
def test_super_strong_anchor_threshold_and_nonempty_title(monkeypatch, score, title, expected_weight, first):
    """85分含等号且法名非空才提权；未获向量印证的超强锚点可以超过单路语义第一名。"""
    rows = [{"id": "core", "title": title, "text": "强词面条款"},
            {"id": "semantic", "title": "其他规范", "text": "泛化语义条款"}]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(0, score)])
    retriever.vectors = [[0.0, -1.0], [0.0, 1.0]]
    hits, mode = asyncio.run(retriever.search("复合问题", top_k=2))
    assert mode == "hybrid" and hits[0]["id"] == first
    by_id = {hit["id"]: hit["retrieval_score"] for hit in hits}
    assert by_id["core"] == pytest.approx(round(expected_weight / 31, 5))


@pytest.mark.parametrize("score, expected_weight, first", [(139.999, 1.35, "semantic"), (140.0, 2.1, "core")])
def test_multiple_phrase_anchor_beats_single_remedy_vector_at_140(monkeypatch, score, expected_weight, first):
    """140含等号时精确词面须超过救济向量1.62；139.999仍保持1.35旧档。"""
    rows = [{"id": "core", "title": "核心规范", "text": "多要件词面条款"},
            {"id": "semantic", "title": "其他规范", "text": "当事人有权请求救济。"}]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(0, score)])
    retriever.vectors = [[0.0, -1.0], [0.0, 1.0]]
    hits, mode = asyncio.run(retriever.search("多个法律问题", top_k=2))
    assert mode == "hybrid" and hits[0]["id"] == first
    by_id = {hit["id"]: hit["retrieval_score"] for hit in hits}
    assert by_id["core"] == pytest.approx(round(expected_weight / 31, 5))
    assert by_id["semantic"] == pytest.approx(round(1.2 * 1.35 / 31, 5))


@pytest.mark.parametrize("title, weight", [("", 1.0), (None, 1.0), ("中华人民共和国工会法", 0.1)])
def test_multiple_phrase_anchor_still_requires_title_and_scope(monkeypatch, title, weight):
    """高于140也不能绕过空法名或专门法范围检查。"""
    rows = [{"id": "core", "title": title, "text": "多要件条款"},
            {"id": "semantic", "title": "其他规范", "text": "普通条款"}]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(0, 150.0)])
    retriever.vectors = [[0.0, -1.0], [0.0, 1.0]]
    hits, mode = asyncio.run(retriever.search("一般纠纷如何处理", top_k=2))
    assert mode == "hybrid"
    by_id = {hit["id"]: hit["retrieval_score"] for hit in hits}
    assert by_id["core"] == pytest.approx(round(weight / 31, 5))


@pytest.mark.parametrize("scores, weights", [((150.0, 145.0, 140.0), (2.1, 2.1, 1.0)),
                                            ((150.0, 100.0, 90.0), (2.1, 1.35, 1.0))])
def test_multi_phrase_and_strong_anchors_share_two_title_slots(monkeypatch, scores, weights):
    """2.1与1.35共用每法两条提权名额，不能为新档另开两条配额。"""
    rows = [{"id": str(i), "title": "核心规范", "text": "词面条款"} for i in range(3)] + [
        {"id": "semantic", "title": "其他规范", "text": "普通条款"}]
    retriever = Retriever(rows, Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: list(enumerate(scores)))
    retriever.vectors = [[0.0, -1.0]] * 3 + [[0.0, 1.0]]
    hits, mode = asyncio.run(retriever.search("复合纠纷", top_k=4))
    assert mode == "hybrid"
    by_id = {hit["id"]: hit["retrieval_score"] for hit in hits}
    for i, weight in enumerate(weights):
        assert by_id[str(i)] == pytest.approx(round(weight / (31 + i), 5))


@pytest.mark.parametrize("multiple_laws", [False, True])
def test_multiple_strong_laws_limit_two_per_title_then_preserve_overflow(monkeypatch, multiple_laws):
    """仅多法名超强锚点触发限流；单法不折叠，超额候选保序回填且空法名不被分组限流。"""
    rows = [{"id": f"a{i}", "title": "规范甲", "text": "强锚点"} for i in range(4)] + [
        {"id": "b", "title": "规范乙", "text": "另一强锚点"},
        {"id": "untitled", "text": "无法名条款"},
    ]
    retriever = Retriever(rows, Embedder())
    lexical = [(0, 100.0), (1, 99.0), (2, 98.0), (3, 97.0),
               (4, 96.0 if multiple_laws else 84.999), (5, 80.0)]
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: lexical)
    retriever.vectors = [[0.0, 1.0]] * 4 + [[0.0, -1.0]] * 2
    hits, mode = asyncio.run(retriever.search("复合问题", top_k=6))
    expected = ["a0", "a1", "b", "untitled", "a2", "a3"] if multiple_laws else ["a0", "a1", "a2", "a3", "b", "untitled"]
    assert mode == "hybrid" and [hit["id"] for hit in hits] == expected
    assert len({hit["id"] for hit in hits}) == 6
    by_id = {hit["id"]: hit["retrieval_score"] for hit in hits}
    assert by_id["a0"] == pytest.approx(round(2.55 / 31, 5))
    assert by_id["a1"] == pytest.approx(round(2.55 / 32, 5))
    assert by_id["a2"] == pytest.approx(round(2.2 / 33, 5))


@pytest.mark.parametrize("dense_state, expected_mode", [
    ("absent", "lexical"), ("below_threshold", "hybrid"), ("failed", "lexical_fallback"),
])
def test_empty_dense_keeps_original_lexical_order_and_scores(monkeypatch, dense_state, expected_mode):
    """向量未就绪、无合格命中或查询失败时，原始词面得分和次序完全保留，不做降权。"""
    class FailedEmbedder(Embedder):
        async def embed(self, texts, *, query=False):
            raise RuntimeError("合成向量失败")

    rows = [{"id": str(i), "title": "同一规范", "text": "条款"} for i in range(3)]
    retriever = Retriever(rows, FailedEmbedder() if dense_state == "failed" else Embedder())
    monkeypatch.setattr(retriever._bm25, "rank", lambda query: [(1, 28.0), (0, 12.0), (2, 10.0)])
    if dense_state != "absent":
        retriever.vectors = [[0.0, -1.0]] * 3
    hits, mode = asyncio.run(retriever.search("咨询处理方式", top_k=3))
    assert mode == expected_mode
    assert [(hit["id"], hit["retrieval_score"]) for hit in hits] == [("1", 28.0), ("0", 12.0), ("2", 10.0)]


def test_corpus_is_real_verified_and_unique():
    """测试语料加载基础属性：ID 必须全局唯一，内容必须带有快照核验范围标注"""
    rows, coverage = load_corpus()
    assert coverage["laws"] >= 2 and coverage["cases"] == 6
    assert len(rows) >= 94
    # 核心断言：所有切片 ID 必须完全唯一，不允许重复条目
    assert len({r["id"] for r in rows}) == len(rows)
    # 必须标明核验范围
    assert all(r["verified_scope"].startswith(("与已保存", "采集官网HTML")) for r in rows)
    # 案例必须标明仅为发布稿摘要，不是判决书全文
    assert any(r["content_scope"] == "RELEASE_SUMMARY_ONLY" for r in rows)


def test_cache_avoids_reembedding_documents_but_not_user_queries(tmp_path):
    """测试向量缓存机制与用户隐私保护：
    1. 相同条文二次加载时直接读取缓存，不重复调用 embedding；
    2. 用户的问题向量实时计算，但绝不能被写入本地磁盘缓存文件。
    """
    async def run():
        rows = [{"id": "a", "text": "工伤保险", "kind": "LAW_ARTICLE"}]
        e = Embedder()
        # 第一次构建索引：调用 embedding 并写入缓存
        first = Retriever(rows, e, tmp_path / "index.json")
        await first.build()
        assert not e.calls[0][1]  # 首次调用不是用户 query
        
        # 第二次加载同一份资料：应当完全命中缓存，调用次数仍为 1
        second = Retriever(rows, e, tmp_path / "index.json")
        await second.build()
        assert len(e.calls) == 1
        
        # 搜索测试：计算用户 query 向量
        hits, mode = await second.search("工伤", top_k=2)
        assert mode == "hybrid" and hits[0]["id"] == "a"
        assert len(e.calls) == 2 and e.calls[-1][1]  # 本次是用户 query
        
        # 隐私安全断言：用户搜索的关键词“工伤”绝对不能存入磁盘缓存
        assert "工伤" not in (tmp_path / "index.json").read_text()

    asyncio.run(run())


def test_corrupt_cache_rebuilt(tmp_path):
    """测试坏缓存自动容灾：当缓存文件损坏或格式非法时，自动重新构建，不崩溃"""
    async def run():
        p = tmp_path / "index.json"
        p.write_text('{"vectors": [[1]]}')  # 伪造损坏缓存
        e = Embedder()
        r = Retriever([{"id": "a", "text": "工伤", "kind": "LAW_ARTICLE"}], e, p)
        await r.build()
        # 缓存损坏后重新生成了一次
        assert len(e.calls) == 1

    asyncio.run(run())


def test_adding_one_article_only_embeds_new_article(tmp_path):
    """测试增量更新：库里已有条文 a，新增条文 b，只对条文 b 发起向量计算，条文 a 复用缓存"""
    async def run():
        e = Embedder()
        path = tmp_path / 'index.json'
        first = Retriever([{'id': 'a', 'text': '工伤'}], e, path)
        await first.build()
        
        # 加入新条文 b
        second = Retriever([{'id': 'a', 'text': '工伤'}, {'id': 'b', 'text': '工资'}], e, path)
        await second.build()
        # 确认第二次实际发给 embedding 的只有新条文“工资”
        assert e.calls[-1][0] == ['工资']

    asyncio.run(run())


def test_failed_build_resumes_saved_batches(tmp_path):
    """测试断点续传：17 条数据在处理完前 16 条后遭遇异常中断，重试时自动从第 17 条断点继续"""
    class Interrupted(Embedder):
        """模拟跑完第一批（16条）后突然报错断电"""
        async def embed(self, texts, *, query=False):
            if self.calls:
                raise RuntimeError('模拟中途失败')
            return await super().embed(texts, query=query)

    async def run():
        rows = [{'id': str(i), 'text': '工伤' + str(i)} for i in range(17)]
        path = tmp_path / 'index.json'
        broken = Retriever(rows, Interrupted(), path)
        await broken.build()
        # 第一批 16 条已通过检查点安全保存
        assert broken.state == 'lexical' and broken.completed == 16
        
        # 换好环境重新启动，自动从断点续跑
        e = Embedder()
        resumed = Retriever(rows, e, path)
        await resumed.build()
        assert resumed.state == 'ready'
        # 确认仅对遗漏的第 17 条（'工伤16'）发起了计算请求
        assert e.calls[0][0] == ['工伤16']

    asyncio.run(run())


def test_document_embedding_text_adds_law_header_without_mutating_row():
    """新格式只改送模文本；rows 里的 text 原样保留，未知格式明确报错。"""
    row = {"id": "a", "title": "测试法", "locator": "第一条 正文提取行L1", "text": "应当支付工资。"}
    original = dict(row)
    assert document_embedding_text(row) == "应当支付工资。"
    assert document_embedding_text(row, DOC_FORMAT_LAW_HEADER_V1) == "《测试法》第一条 正文提取行L1\n应当支付工资。"
    assert row == original
    with pytest.raises(ValueError):
        document_embedding_text(row, "unknown_format")


def test_doc_format_default_cache_paths_are_separate():
    """plain 默认仍指向旧 embeddings.json；新格式用独立文件，避免覆盖旧向量。"""
    rows = [{"id": "a", "text": "工资"}]
    plain = Retriever(rows, Embedder())
    new = Retriever(rows, Embedder(), doc_format=DOC_FORMAT_LAW_HEADER_V1)
    assert plain.cache_path == ROOT / "data/local_index/embeddings.json"
    assert new.cache_path.name == "embeddings.law_header_v1.json"
    assert new.cache_path != plain.cache_path


def test_new_doc_format_embeds_header_text_and_marks_cache(tmp_path):
    """新格式实际送模文本带法名条号，缓存和 sync 都带格式标记；原文一字不变。"""
    class Store:
        def __init__(self):
            self.args = None

        async def sync(self, rows, vectors, model_id, doc_format="plain"):
            self.args = (rows, vectors, model_id, doc_format)

    async def run():
        rows = [{"id": "a", "title": "测试法", "locator": "第一条 正文提取行L1",
                 "text": "应当支付工资。", "kind": "LAW_ARTICLE"}]
        embedder, store = Embedder(), Store()
        path = tmp_path / "index.json"
        retriever = Retriever(rows, embedder, path, store=store, doc_format=DOC_FORMAT_LAW_HEADER_V1)
        await retriever.build()

        assert retriever.state == "ready"
        assert embedder.calls[0][0] == ["《测试法》第一条 正文提取行L1\n应当支付工资。"]
        assert rows[0]["text"] == "应当支付工资。"
        assert store.args[3] == DOC_FORMAT_LAW_HEADER_V1
        cached = json.loads(path.read_text(encoding="utf-8"))
        assert cached["doc_format"] == DOC_FORMAT_LAW_HEADER_V1
        assert cached["entries"]

    asyncio.run(run())


def test_new_doc_format_never_reuses_plain_cache(tmp_path):
    """同一缓存路径下，新格式发现缺少格式标记就重算，不能直接复用旧正文向量。"""
    async def run():
        rows = [{"id": "a", "title": "测试法", "locator": "第一条", "text": "工伤待遇"}]
        path = tmp_path / "index.json"
        embedder = Embedder()
        plain = Retriever(rows, embedder, path)
        await plain.build()
        assert len(embedder.calls) == 1

        new = Retriever(rows, embedder, path, doc_format=DOC_FORMAT_LAW_HEADER_V1)
        await new.build()
        assert len(embedder.calls) == 2
        assert embedder.calls[-1][0] == ["《测试法》第一条\n工伤待遇"]

    asyncio.run(run())

