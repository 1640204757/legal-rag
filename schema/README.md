# OpenLawGuide Schema v0

版本：0.3.0
日期：2026-10-19
状态：草案。仅定义法规/案例的最小机器契约与正反样例，**不包含业务代码、不包含真实数据**。字段名、枚举和必填集可能在首批验证后调整。
0.2.0 修正：`model_summary` 改为可选/可为 null；案例身份增加 `verification_status`，`verified_at` 允许 null；法规 `repeal_date` 明确为“字段必须出现，null 表示日期未知”。
0.3.0 修正：`retrieved_at`（采集时间）与 `imported_at`（导入时间）分离，增加 `capture_manifest_ref`；新增 `citation_eligibility` 四要件 + verdict，明确“仅逐字核验通过不足以标法定依据”。

## 1. 文件

| 文件 | 用途 |
| --- | --- |
| `legal_document.schema.json` | 法规/司法解释文档与条款结构 |
| `case_document.schema.json` | 案例文档、身份、内容分区、官方要旨、引用卡片 |
| `examples/legal_document.valid.json` | 有效法规样例（虚构） |
| `examples/legal_document.invalid.json` | 无效法规样例（故意缺字段/非法枚举） |
| `examples/case_document.valid.json` | 有效案例样例（虚构，含多身份重叠，其中一个身份未核验） |
| `examples/case_document.null_summary.valid.json` | 有效案例样例（虚构，`model_summary: null`） |
| `examples/case_document.omitted_summary.valid.json` | 有效案例样例（虚构，完全缺省 `model_summary`） |
| `examples/case_document.invalid.json` | 无效案例样例（故意缺字段/非法日期/身份缺核验状态） |
| `examples/case_document.invalid_unverified_effect.json` | 无效案例样例（未核验身份却给出 `SHOULD_REFER`） |
| `validation_report_20261019.txt` | 本轮完整校验的原始输出（jsonschema 4.26.0 + FormatChecker） |

Schema 采用 JSON Schema Draft 2020-12；`additionalProperties: false`，新增字段需升版本。

## 2. 首批必填子集

### 2.1 法规 `legal_document`

必填：`schema_version`、`doc_id`、`version_id`、`title`、`doc_type`、`issuing_body`、`jurisdiction`、`validity_status`、`repeal_date`、`authority_roles`、`official_url`、`retrieved_at`、`articles`。

其余可空/可缺省：文号、公布/施行/修订日期、效力依据、制定/授权依据、过渡规定、版本关系、核验时间、校验值、再利用依据、条款层级中的父级/引用/生效区间。

关键规则：

- `doc_id` 跨版本稳定；`version_id` 区分同一资料的不同版本；**历史版本独立保存，不因当前失效被排除**。
- `retrieved_at` 是**实际采集时间**，来自采集记录，导入时不得覆盖；`imported_at` 单独记录解析时间；`capture_manifest_ref` 指向采集记录。
- `repeal_date` 字段必须出现；`null` 表示**日期未知**，不等于未废止。`validity_status=REPEALED` 时也允许 `null`；不允许因当前失效而丢弃历史版本。
- `authority_roles` 表示依据资格：`ADJUDICATIVE_BASIS`（裁判依据）、`REFERENCE`（参照）、`REASONING_REFERENCE`（说理参考）、`PROCEDURAL_GUIDE`（程序指引）、`UNKNOWN`。**仅表示文件性质上的资格，不等于本案可直接引用。**
- `citation_eligibility` 记录引用资格核验结果：`verified_verbatim`（逐字核验）、`version_confirmed`（版本适用）、`authority_eligible`（依据资格）、`facts_matched`（事实要件）；只有 `verdict=ELIGIBLE` 且四项全 true 才可为真。**仅逐字核验通过不足以标“法定依据”。**
- 条款层级支持 `PART/CHAPTER/SECTION/ARTICLE/PARAGRAPH/ITEM/SUBITEM`，无编号层级允许 `article_no: null`。

### 2.2 案例 `case_document`

必填：`schema_version`、`case_id`、`title`、`court`、`level`、`procedure`、`judgment_type`、`judgment_date`、`cause_of_action_raw`、`identities`、`sections`、`official_gists`、`applicable_law_versions`、`related_cases`、`effective_status`、`public_status`、`collection_policy`、`source_url`、`source_body`、`retrieved_at`。

可选/可空：`model_summary`（缺省或 `null` 表示尚未生成；存在时必须带 `model_version` 与 `review_status`）、`citation_card`、案号、案例库编号、发布日期、地域、案由标准化值、罪名、核验时间、校验值。

关键规则：

- `identities` 多标签：指导性案例、案例库参考案例、公报、典型案例、地方参考性案例、最高检案例、普通文书、未知。每项必须记录 `source`、`verification_status`、`normative_effect`；`verified_at` 可缺省或为 `null`。
- `verification_status=VERIFIED` 时必须有具体 `verified_at`；`UNVERIFIED`/`UNKNOWN` 时 `normative_effect` 只能为 `UNKNOWN`，且**不得在界面展示为“已确认身份/官方确认”**。
- `sections` 分区：`CLAIM`、`DEFENSE`、`COURT_FOUND_FACTS`、`ISSUES`、`EVIDENCE`、`PROCEDURE`、`REASONING`、`APPLIED_LAW`、`RESULT`、`UNCLASSIFIED`。每段必须带 `locator`（原文定位）、`extract_method`、`review_status`、`subject_label`；抽取失败用 `UNCLASSIFIED`，不得强行归类。
- `official_gists` 按来源分类型：`JUDICIAL_POINTS`（裁判要点）、`JUDICIAL_GIST`（裁判要旨）、`JUDICIAL_ABSTRACT`（裁判摘要）、`TYPICAL_SIGNIFICANCE`（典型意义）；不得合并为单一“官方要旨”。
- `model_summary` 不是必填；不得为满足 Schema 而编造模型版本或摘要。展示层不得使用“法院认为/法院认定”。
- `applicable_law_versions` 记录案例裁判时的法规版本与条号，用于类案可比性。
- `related_cases` 记录同案不同审级、串案、上诉/再审关系。
- `public_status`（是否官方公开）、`collection_policy`（项目采集政策）与法律限制分开；`POLICY_EXCLUDED` 不等于法律禁令。
- `citation_card.coverage_note` 必须能说明案例检索覆盖范围（如仅入库/已发布案例）。

## 3. 未知与缺失

- 必填字段必须出现；未知值用 `null`（字段允许 null 时）或枚举 `UNKNOWN`。
- 不允许用空字符串、编造日期或推测值填充未知。
- 来源无法核验时，身份用 `verification_status: UNVERIFIED/UNKNOWN`、`verified_at: null`，并在来源目录中记“未知”；不得展示为已确认身份。
- `model_summary` 可为 `null`（尚未生成）；`repeal_date: null` 表示废止日期未知；`verified_at: null` 表示身份未核验或时间未知。三者都不等于“无此字段/未废止/无身份”。
- `content_hash`、`verified_at` 允许为 `null`。

## 4. 正反样例

- 有效样例全部使用 `example.invalid` 域名与虚构机关/案号，明确标注“虚构”。
- 无效样例分别演示：缺少必填字段、非法枚举、非法日期、身份缺少规范效果/核验状态、`articles` 为空、未核验身份却给出已确认效果。
- 有效样例同时覆盖：`model_summary` 为对象、`model_summary: null`、完全缺省 `model_summary`、已核验身份、未核验身份、多身份重叠、历史版本字段。
- 样例只用于验证结构，不用于法律结论；不得把样例文本当作真实法规或案例。

## 5. 校验

已使用 `jsonschema` 4.26.0 完成完整校验：

- `Draft202012Validator.check_schema()` 对两份 Schema 本身校验通过。
- 显式启用 `FormatChecker()`，`date`、`date-time`、`uri` 等 format 检查实际参与校验，而不是只检查结构。
- 7 个样例结果：
  - 有效：`legal_document.valid.json`、`case_document.valid.json`、`case_document.null_summary.valid.json`、`case_document.omitted_summary.valid.json` 全部通过；
  - 无效：`legal_document.invalid.json`、`case_document.invalid.json`、`case_document.invalid_unverified_effect.json` 全部被拒绝。
- 关键负例已验证：`2025-13-45` 被 `date` format 拒绝；未核验身份给 `SHOULD_REFER` 被 `allOf/if-then` 规则拒绝；缺少 `repeal_date` 被 `required` 拒绝。

复现方式：使用任一 Draft 2020-12 校验器，并显式启用 format 检查；例如 Python：

```python
from jsonschema import Draft202012Validator, FormatChecker
validator = Draft202012Validator(schema, format_checker=FormatChecker())
errors = list(validator.iter_errors(instance))
```

只跑两个有效、两个无效样例不足以宣称契约验证充分；本轮已扩充到 7 个样例，覆盖可选模型摘要、未核验身份、废止日期未知、非法日期与非法枚举。原始输出见 `validation_report_20261019.txt`。

## 6. 版本与兼容

- 当前 `schema_version: "0.3.0"`；`additionalProperties: false` 意味着新增/改名均属破坏性变更，需升版本并更新样例。
- 首批验证后可能调整字段；调整时保留旧样例与迁移说明。
- 本 Schema 不包含向量、索引、检索评分等实现字段；这些在实现阶段另定，不混入数据契约。

## 7. 与文档的关系

- 需求级字段说明见 `legal_case_rag_requirements_review.md` §4。
- 案例契约与引用标签见 `detailed_design_spec.md` §4.3。
- 来源与未知标记见 `source_catalog_v1.md`。
