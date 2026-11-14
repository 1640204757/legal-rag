# 云端 RAG 对照评测：上传与运行说明

日期：2026-10-19
目标：在干净 Qwen3-8B 基座上，对同一批留出题比较“不提供检索资料”与“提供检索资料”两种回答。
**本包不含模型权重、不含 LoRA。** 上一轮的 smoke LoRA 仅验证了训练流程，不是法律模型，本轮不加载。

## 1. 包内文件

| 文件 | 说明 |
| --- | --- |
| `package/eval_tasks.json` | 12 题留出题 + 人工核对材料（M1、M2…）+ 覆盖缺口 + 期望行为 |
| `package/snapshots.json` | 3 份原文快照全文，供引文逐字核验 |
| `run_rag_eval.py` | 自包含评测脚本（仅标准库 + 可选 torch/transformers） |
| `README.md` | 本文件 |

材料来源为人工核对映射（`data/eval/holdout_gold_materials_v1.json`），不是自动检索结果。原因见 §6。

## 2. 上传

把 `cloud/` 目录整体上传到 Colab（或任一有 GPU 的机器），建议放在 `/content/rag_eval/`：

```
rag_eval/
├── run_rag_eval.py
└── package/
    ├── eval_tasks.json
    ├── snapshots.json
```

也可以只上传三个文件（脚本 + 两个 JSON），放到同一目录下并调整路径参数。

### 2.1 两套系统提示词（有材料 / 无材料）

脚本按本题**是否提供了材料**自动选择提示词分支：

| 分支 | 适用 | 关键约束 |
| --- | --- | --- |
| `SYSTEM_PROMPT_WITH_MATERIALS` | 有 M1… 材料 | 只能引用材料内编号；引用格式固定；**不在提示词里给可照抄的引文示例** |
| `SYSTEM_PROMPT_NO_MATERIALS` | 无材料 | 可引用编号集合为**空集**；禁止任何形如 `[M数字]` 的标记；禁止用“示例引用”变通保留编号 |

> 第二轮实测发现：旧提示词里写了 `原文：“……”[M1]` 作为格式示例，模型在无材料时**直接照抄成“示例引用”**（baseline H02）。现在无材料分支已移除一切编号示例，并由 `test_no_material_prompt_has_no_copyable_example` 守住。

### 2.2 生成后后处理（放行判定）

每条答案生成后额外做一次后处理，检测五类问题：

| 类型 | 含义 |
| --- | --- |
| `NO_MATERIAL_HAS_CITATION` | 本轮无材料，却出现编号 |
| `UNKNOWN_ID` | 出现未提供的编号 |
| `FAKE_EXAMPLE_CITATION` | 用“示例引用/此处为示例”保留编号（伪引用） |
| `SELF_CONTRADICTION` | 处理方向用确定表述，不确定性段又说信息不足 |
| `CITATION_IN_UNCERTAINTY` | 在“不确定性”段挂编号 |

**失败答案是原样保留的**：`raw_answer == answer`、`modified=false`，不删除编号、不改写内容后重新判定通过。只要命中任一阻断项，`displayable_as_verified=false`，该答案不得作为“核验通过”的结果展示。

## 3. 先干跑（不需要 GPU）

```bash
cd /content/rag_eval
python run_rag_eval.py --tasks package/eval_tasks.json --snapshots package/snapshots.json \
  --mock --limit 2 --out mock_results.json
```

预期：`baseline` 的 `cited_rate=0.0`；`rag` 组出现 `VERBATIM_OK`，`citation_state_counts` 正常分布。
这只验证流程与引用检查，**不代表模型能力**。

## 3.1 首轮实测已暴露的失败点（H01/H02，已转为调试案例）

2026-10-19 首轮云端两题结果存于 `artifacts/cloud-runs/results_qwen3_8b_first2.json`。

| 题 | 首轮判定 | 问题 | 本轮处理 |
| --- | --- | --- | --- |
| H01 | 旧检查器 **PASS** | 无逐字引文却通过；`[ M2 ]` 未识别；材料（第六条/十八条）不是欠薪依据，诱导模型跑到解除与二倍工资 | 四态状态（判 ID_ONLY/FAIL）；编号空格兼容；**H01 改为不提供材料**，专测承认缺口 |
| H02 | **FAIL(UNBOUND)** | 长引文其实在 M1 中，因编号隔着解释而超窗；强调引号被当成待核验引文 | 段落绑定 + 强调引号识别；改判 VERBATIM_OK 并给格式警告 |

**这两题已用于调试，不再是“未见过的留出题”。** 修正后须先复测这两题，通过后才跑其余题；最终报告需把这两题单独标注为“已见题”。

### 3.2 第二轮实测暴露的失败点（已修正）

结果存于 `artifacts/cloud-runs/results_qwen3_8b_seen2_v2.json`。

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| rag H01：无材料却写“现有材料不足以支持确定结论[M1]” | 提示词全篇是同一套示例，模型照抄编号 | 无材料独立分支 + 后处理 `NO_MATERIAL_HAS_CITATION` |
| baseline H02：`原文：“…应当认定为工伤。”[M1]（注：此处为示例引用）` | **提示词里的 `原文：“……”[M1]` 被直接照抄** | 移除提示词中的编号示例；后处理识别 `FAKE_EXAMPLE_CITATION` |
| rag H02：先说“可认定为工伤”，又说“不足以支持确定结论” | 缺条件性分析约束 | 后处理 `SELF_CONTRADICTION`；提示词要求统一条件化表述 |
| rag H02：未利用 M2 提期限 | 未要求按适用事实提示程序条款 | 提示词要求在追问/不确定性中提示，但只提示适用条件、不机械抄全部期限 |

## 4. 正式跑（需要 GPU）

```bash
pip install -q transformers accelerate bitsandbytes

python run_rag_eval.py \
  --tasks package/eval_tasks.json \
  --snapshots package/snapshots.json \
  --model Qwen/Qwen3-8B \
  --modes baseline,rag \
  --out results_qwen3_8b.json
```

说明：

- 默认加载 4bit（`--no-4bit` 可关闭）；`device_map="auto"`。
- 生成参数：贪心解码（`do_sample=False`），`max_new_tokens=900`，`enable_thinking=False`。
- 同一脚本内先跑 baseline 再跑 rag，两组题面与材料编号完全一致。
- 结果写入 `results_qwen3_8b.json`，含逐题答案、检查结果、耗时。

若显存不足，可先限题数验证：

```bash
python run_rag_eval.py ... --limit 4
```

## 5. 输出指标说明

| 指标 | 含义 | 注意 |
| --- | --- | --- |
| `cited_rate` | 有编号引用的答案比例 | baseline 应接近 0；有引用不等于引用合格 |
| `structure_ok_rate` | 五段结构完整比例 | 结构合格 ≠ 内容正确 |
| `citation_state_counts` | 四态分布（见 §5.1） | **主指标看这个** |
| `verbatim_ok_rate` | `VERBATIM_OK` 占比 | 真正的“逐字引用合格率” |
| `citation_fail` / `citation_id_only` | FAIL / 仅编号的题数 | ID_ONLY 不能算通过 |
| `unknown_id_answers` | 出现未提供编号的答案数 | 应恒为 0 |
| `quote_verified_rate` | 引文与标注材料、快照同时一致的比例 | 只证明“抄得准且抄对地方” |
| `format_warnings` | 编号未紧邻引文的次数 | 提示模型改用固定引用格式 |
| `postprocess_blocked` | 被后处理阻断的答案数 | **主指标之一** |
| `displayable_as_verified` | 可作“核验通过”展示的答案数 | 引用检查与后处理都放行才算 |
| `postprocess_issue_types` | 阻断问题类型分布 | 见 §2.2 |
| `latency_p50_s` / `latency_max_s` | 单题生成耗时 | 用于比较两组开销 |

**这些指标都不衡量法律结论正确性。** 引用通过只说明“可核对”，不说明“适用正确”。

### 5.1 引用检查：四种状态（不得合并成单一通过率）

| 状态 | 含义 | 判定 |
| --- | --- | --- |
| `NO_CITATION` | 回答中没有任何编号引用 | 提供了材料时判 **FAIL** |
| `ID_ONLY` | 有编号，但没有任何逐字引文 | 判 **ID_ONLY**（不是 PASS） |
| `VERBATIM_OK` | 至少一处逐字引文，且全部通过绑定与快照核验 | PASS |
| `VERBATIM_PARTIAL` | 部分引文通过、部分失败 | **FAIL** |
| `VERBATIM_FAIL` | 全部引文失败 | **FAIL** |

`ID_ONLY` 是独立状态：只提到“依据 M1”，不等于引用了原文。首轮云端 H01 就是这种形态，旧检查器却给了 PASS。

### 5.2 引文绑定规则

1. **显式格式**（推荐）：`原文：“……”[M1]`，编号紧贴引号，允许 `[ M1 ]` 这种带空格写法。
2. **紧邻**：引文后 30 字符内、且间隔不超过 8 字符的编号。
3. **段落唯一编号**：格式不规范但该段落只有一个编号时，按段落绑定，同时给 `format_warnings`。
4. **段落多编号** → `AMBIGUOUS_BINDING`，判失败，不替模型猜归属。

短于 8 字且未绑定编号的引号（如“非本人主要责任”）识别为**行文强调**，记入 `emphasis_quotes`，不计入待核验引文。

### 5.3 四项独立判定

| 判定 | 含义 |
| --- | --- |
| `bound_ok` | 引文出现在所标注材料的正文里，且能在对应快照中逐字找到 |
| `mismatch` | 引文实际来自另一份提供的材料，却标成了这个编号 |
| `snapshot_only` | 引文在同份快照里，但不在提供给模型的该材料正文范围内（引用未提供的条款） |
| `other_doc` / `not_found` / `unbound` / `unknown_id` / `snapshot_mismatch` / `ambiguous_binding` | 其他失败类型 |

2026-10-19 两次修正：

1. 此前检查器直接在**整份快照**里找引文，找到就算通过，因此“引用 M1、却抄了同快照中另一条未提供的条款”也能过 → 改为绑定标注编号 + 材料正文匹配。
2. 首轮云端实测又暴露两点：`[ M2 ]` 带空格识别不到；完全没引文的回答被判 PASS → 增加编号空格兼容、四态状态、强调引号识别、段落绑定与歧义判定。

本地与云端检查器现在是**同一份源码**（同步生成），并由测试锁定结果一致。

## 6. 为什么材料是人工核对的

首版曾用本项目词面检索自动选材，人工逐题核对后发现明显错误：

- H01（欠薪）被配了转包条款；
- H03（违法解除）被配了竞业限制条款；
- H10（未缴工伤保险）的直接依据应是条例第六十二条，却被挤到第 4 位而未入选。

原因：纯词面/IDF 检索对长中文问句和口语表达不可靠（`用人单位`、`劳动者` 这类高频词主导排序）。
把不相关材料交给模型，会让“RAG 是否带来改善”的结论失真，因此改为：

- **默认材料 = 人工核对映射**（可审计，依据写在 `selection_rationale`）；
- **自动检索结果作为对照记录**（`--auto-for-compare` 时写入 `auto_retrieval_compare`），用于后续改进检索，不参与本轮判断。

## 7. 已知限制

1. 材料来自离线快照，未做在线逐字复核。
2. 语料仅劳动领域（2 部法规 + 6 个案例发布稿摘要），刑事/民事/行政/程序为 0。
3. 案例材料仅官方发布稿摘要，`content_scope=RELEASE_SUMMARY_ONLY`，不是判决书全文。
4. H01/H03 的材料“相关但不直接”，已在 `gaps` 明确缺失项；这是诚实标注，不是完备依据。
5. 留出题 12 道，样本小；结论只能作为方向性证据。
6. 提示词与评测指标由同一人编写，存在设计偏差风险；未见独立复核。

## 8. 结果怎么读

先看三件事，再决定是否微调：

1. **引用**：rag 组 `unknown_id_answers` 必须为 0；`quote_verified_rate` 高说明模型会照抄且抄对位置；再看 `breakdown` 里有无 `mismatch`（张冠李戴）或 `snapshot_only`（引用未提供条款）。
2. **追问与缺口**：baseline 往往会给笼统建议；rag 组是否更愿意说“材料不足以支持确定结论”。
3. **耗时**：rag 输入更长，记录 `latency_p50_s` 差异是否可接受。

### 8.1 归因顺序（重要修正）

如果**人工选材组（rag）没有明显改善，不要直接归因于检索**。按顺序排查：

1. **材料充分性**：本题语料是否真的能给到直接依据？`gaps` 里写的缺失是否足以让任何模型都答不实？
2. **提示词**：五段结构、引用格式、缺口要求是否与模型能力匹配？模型是否根本没学会这套输出约定？
3. **模型**：基座是否有能力按指令引用？换更强模型是否有改善？
4. **评测方式**：`expect` 与自动检查是否误判？（例如把合理的追问算作不合格）
5. **最后才是检索**：以上四项排除后，才考虑材料选错/漏选。

原因：本轮的 rag 组材料是**人工核对**的，不经过自动检索。它只能验证“给定材料后模型如何回答”，**不能证明实际 RAG 系统已可用**。自动检索目前仍不可用（见 §6），修好后应增加第三组“自动检索材料”对照。

微调的依据也不是“rag 没用”，而是错误类型：
- 资料缺失 → 补资料；
- 召回错误 → 修检索；
- 模型仍不会追问/引用/说明条件 → 才准备正式微调数据。
