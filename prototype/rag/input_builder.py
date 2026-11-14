"""RAG 模型输入包与回答结构。

输入包固定字段（缺一不可）：
- 用户问题
- 必要历史事实（来自多轮对话，标注来源与置信）
- 检索原文（带编号、来源、版本、核验范围）
- 覆盖缺口

回答结构固定五段（顺序固定，标题必须出现）：
追问 / 处理方向 / 证据清单 / 引用 / 不确定性

边界：
- 只允许引用本次提供的材料编号，不得引用外部编号；
- 结构通过 ≠ 法律正确，仅为可核对的最低要求；
- 覆盖缺口必须显式写出，不能用“一般来说”掩盖资料不足。

依赖：仅标准库。
"""

from __future__ import annotations

from dataclasses import dataclass, field

ANSWER_SECTIONS = ("追问", "处理方向", "证据清单", "引用", "不确定性")

# ---------------------------------------------------------------- 通用规则（两分支共享）

_COMMON_RULES = """【回答范围规则】
1. 围绕用户实际提出的诉求作答。不要主动扩展到用户没问的争点；
   若认为其他争点可能相关，只能在“追问”里用**一个问题**询问。
2. **追问要具体、可回答**：问事实（时间、主体、是否书面、金额、是否已申请等），
   不要问“是否存在合法劳动关系”这类需要法律判断的问题。每次追问不超过 3 项。
3. **条件不齐时只能做条件性分析**：写成“如果……则可能……；若……则……”，不要在同一回答里
   先下确定结论、后面又说信息不足。**若只是先陈述一般规则、再说明本案事实不足，不算矛盾。**
4. 严格区分“必要条件”与“充分条件”。
5. 不得给出确定金额、胜诉率或确定倒计时。
6. 涉及期限时，必须写清三件事：**谁**（申请主体）、**从何时起算**（触发事实）、**依据哪条材料**；
   若材料未提供该期限，明确写“材料未包含期限规定”。不要给出没有主体与起算条件的孤立天数。
7. 结论所需事实或依据不足时，在“不确定性”写明缺什么；不得凭常识补全法律规则。

【输出格式】
正文使用以下四个二级标题，顺序固定：

## 追问
## 处理方向
## 证据清单
## 不确定性

**不要抄写条文原文**。引用原文由系统根据你声明的编号自动附列。
正文末尾单独一行声明你用到的材料编号：

USED_IDS: M1, M2

若没有使用任何材料，写 `USED_IDS: 无`。
正文中可以用“依据 M1”这样的简写指代材料，但不要自己抄条文、不要编造编号。

输出语言为中文，不要输出与本结构无关的开场白或免责声明堆砌。"""

# ---------------------------------------------------------------- 分支 A：有材料

SYSTEM_PROMPT_WITH_MATERIALS = """你是面向普通人的法律事务辅助助手。本轮**提供了检索材料**，请按以下规则回答：

【材料使用规则】
1. 只能使用【检索材料】里出现过的编号（M1、M2……）；不得引用材料之外的编号、条文或案例。
2. 不需要抄写条文原文：系统会根据 `USED_IDS` 自动附列对应原文。
   你只需说明“依据 M1 的什么规则”，不要自己复述条文，更不要编造引文。
3. 材料标注“仅发布稿摘要”时，不得当作判决书全文，也不得据此推断裁判文书细节。
4. 材料里若有与处理路径相关的程序条款（如申请期限），按“主体 + 起算 + 依据编号”提示核对；
   不要机械抄出全部期限，也不要在材料没有期限规定时自行补一个。

""" + _COMMON_RULES

# ---------------------------------------------------------------- 分支 B：无材料

SYSTEM_PROMPT_NO_MATERIALS = """你是面向普通人的法律事务辅助助手。本轮**没有提供任何检索材料**，请按以下规则回答：

【无材料规则（最重要）】
1. 本次可引用编号集合为**空集**。回答中**不得出现任何形如方括号加字母 M 加数字的编号标记**，
   无论是紧贴引号、带空格、还是放在句末，一律不允许。
2. 不得凭记忆复述法条原文再附编号；可以说明“一般需要核对相关规定”，但不能伪装成已核验引用。
3. 不得使用“示例引用”“此处为示例”“仅作示例”等变通写法保留编号。
4. 仍要给出有用内容：追问关键事实、说明可考虑的处理方向、列出应准备的证据、写明缺什么。
5. 末尾写 `USED_IDS: 无`。

""" + _COMMON_RULES

# 兼容旧接口
SYSTEM_PROMPT = SYSTEM_PROMPT_WITH_MATERIALS

# 输出结构（用于结构检查）：模型不再单独输出“引用”段
ANSWER_SECTIONS = ("追问", "处理方向", "证据清单", "不确定性")


@dataclass
class Material:
    """一条检索材料。"""

    material_id: str
    kind: str  # LAW_ARTICLE / CASE_RELEASE_EXCERPT
    title: str
    locator: str
    version_id: str
    source_url: str
    text: str
    verified_scope: str  # 例如“与已保存原文快照一致”
    content_scope: str = "FULL_TEXT"  # FULL_TEXT / RELEASE_SUMMARY_ONLY
    snapshot_id: str = ""
    # 由 answer_assembly.split_source_excerpts() 切好的完整合法句，供模型按 M1#1 选号。
    # 为空时 render() 与改造前逐字节一致；非空时标记内联在句子边界，去掉标记即可还原原文。
    excerpts: list[str] = field(default_factory=list)

    def render(self) -> str:
        """把当前对象格式化为模型输入文本，保留其来源或核验范围，不生成新事实。"""
        scope_note = ""
        if self.content_scope == "RELEASE_SUMMARY_ONLY":
            scope_note = "\n注意：本条为官方发布稿摘要，不是判决书全文，不得当作全文引用。"
        if self.excerpts:
            # 只在句首插编号标记；片段本身直接拼回，去掉标记后仍是材料原文。
            body = "".join(f"〔{self.material_id}#{i}〕{text}"
                           for i, text in enumerate(self.excerpts, 1))
        else:
            body = self.text
        return (
            f"[{self.material_id}] {self.kind}｜{self.title}｜{self.locator}｜版本 {self.version_id}\n"
            f"来源：{self.source_url}\n"
            f"核验范围：{self.verified_scope}{scope_note}\n"
            f"原文：{body}"
        )


@dataclass
class Fact:
    """一条必要历史事实。"""

    name: str
    value: str
    source: str  # 用户陈述 / 材料记载 / 待确认假设
    note: str = ""

    def render(self) -> str:
        """把当前对象格式化为模型输入文本，保留其来源或核验范围，不生成新事实。"""
        extra = f"（{self.note}）" if self.note else ""
        return f"- {self.name}：{self.value}｜来源：{self.source}{extra}"


@dataclass
class RagInput:
    question: str
    facts: list[Fact] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    coverage_note: str = ""
    # 远期用户原话片段，只作“前情陈述”供模型对照，不是法律依据。
    recalled: list[str] = field(default_factory=list)

    def render_user_prompt(self) -> str:
        """组织本轮问题、事实、材料与缺口，供模型读取；保留各材料身份和核验范围。"""
        parts = ["【用户问题】", self.question.strip(), ""]

        parts.append("【已知事实】")
        if self.facts:
            parts.extend(f.render() for f in self.facts)
        else:
            parts.append("- 无（用户仅提供了问题本身）")
        parts.append("")

        # 只有回查命中时才追加；为空时输出与改造前逐字节一致（单轮钉扎依赖这一点）。
        if self.recalled:
            parts.extend(self.recalled)
            parts.append("")

        parts.append("【检索材料】")
        if self.materials:
            for m in self.materials:
                parts.append(m.render())
                parts.append("")
        else:
            parts.append("（本次无检索材料）")
            parts.append("")

        parts.append("【覆盖缺口】")
        if self.gaps:
            parts.extend(f"- {g}" for g in self.gaps)
        else:
            parts.append("- 无已知缺口")
        parts.append("")

        if self.coverage_note:
            parts.append(f"【覆盖声明】{self.coverage_note}")
            parts.append("")

        if self.materials:
            parts.append("请按系统要求的结构作答，只引用上方材料的编号。")
        else:
            parts.append("请按系统要求的结构作答。注意：本轮没有材料，回答中不得出现任何 [M…] 编号。")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """将输入包转换为普通字典，保留来源、材料和覆盖声明字段。"""
        return {
            "question": self.question,
            "facts": [f.__dict__ for f in self.facts],
            "materials": [m.__dict__ for m in self.materials],
            "gaps": list(self.gaps),
            "coverage_note": self.coverage_note,
            "recalled": list(self.recalled),
        }


def select_system_prompt(materials: list) -> str:
    """按是否有材料选择系统提示词分支。无材料分支禁止一切编号。"""
    return SYSTEM_PROMPT_WITH_MATERIALS if materials else SYSTEM_PROMPT_NO_MATERIALS


def build_messages(rag_input: RagInput) -> list[dict]:
    """根据本轮有无检索材料选择提示词并组装输入，不在此步骤调用模型。"""
    return [
        {"role": "system", "content": select_system_prompt(rag_input.materials)},
        {"role": "user", "content": rag_input.render_user_prompt()},
    ]


def build_ablation_messages(rag_input: RagInput) -> list[dict]:
    """对照组：不提供检索材料，只给问题与事实。用于测量 RAG 带来的差异。"""
    stripped = RagInput(
        question=rag_input.question,
        facts=rag_input.facts,
        materials=[],
        gaps=["本次未提供任何检索材料"],
        coverage_note="对照组：无检索材料",
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_NO_MATERIALS},
        {"role": "user", "content": stripped.render_user_prompt()},
    ]
