# 真实端到端评测与验收流水线

## 阅读顺序与职责

1. `data/eval/retrieval_eval_v2.json`：40题及14题独立防御样例。36题覆盖民法合同/借贷/侵权、劳动人事、公司、消保、教育与程序；4题是特定境外法律或地方细则缺口。
2. `retrieval_eval_v2.frozen.json`：原始字节SHA256。固化后不按分数改题，旧v1及原型评测保留原样。
3. `local_rag/evaluation.py`：分母、三元组、原始/最终显式引用及保守边界评分。评分不是法律结论正确性审查。
4. `local_rag/eval_runtime.py`：只读初始化当前索引，真实网络传输观测、生产ChatService逐题调度、独立防御注入及报告。
5. `scripts/run_rag_eval.py`：命令入口。真实链路为`ChatService → 原Retriever.search → Ollama查询向量 → Qdrant当前corpus查询 → ChatService预算筛选 → 原OllamaClient.chat → 原postprocess及期限检查`。

评测器不将预期法条传给模型或检索，不重写提示词，不调排序或提高token预算；不修生产拦截器来迎合本轮。当前代码观测真实Top4和实际模型输入材料，并保留全部Qdrant响应payload。

## 怎样运行

在项目根目录使用`.local-python-path.txt`记录的解释器。设置：

```text
PYTHONPATH=.;.local-deps
PYTHONIOENCODING=utf-8
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

先完整回归，再启动评测：

```text
python -m pytest tests_local prototype/tests -q --rootdir .
python scripts/run_rag_eval.py
python scripts/annotate_rag_eval_acceptance.py
python scripts/verify_rag_eval_artifacts.py
```

需要本机16333的Qdrant和11434的Ollama，既有资料向量缓存必须有效，问答模型及embedding必须就绪。初始化只读核对694部规范、36,176段、模型与内容指纹以及corpus精确点数；传输层拒绝写库方法。失败时不会自动重算、清库或切换到云端。

以上是首次评测的旧语料边界。2026-10-21核心扩充后当前库为867部规范、46,246段；新库复测入口保留题集中的旧corpus标识，另从已完成扩充的`summary.json`核对新corpus指纹、覆盖数及无待处理/运行worker，再真实核验Qdrant精确点数和缓存。冻结预期的法律、版本、条号、正文哈希仍逐项核对，不修改旧题、预期或历史分数。

新库命令：`python scripts/run_rag_eval.py --output artifacts/local-rag/eval_report_20261021_expansion.json`。逐题`token_usage`来自Ollama终止包的真实输入/输出计数；未返回计数时明确为null，不估算。`scripts/summarize_rag_expansion.py`只读对比三轮成绩，并沿用上轮严格安全降级补查，输出`docs/reports/RAG_EXPANSION_EVAL_20261021.md`及同名新报告旁的comparison.json。新库与期限拦截同时变化，不据此宣称扩充的单独因果收益。

本轮已真实执行：40/40完整，Doc_Hit@4为28/36（77.8%），Q38“两周”被阻断，最终错误引用0/40、严格降级2/4，结论部分通过。逐题耗时合计1168.03秒；输入45,985、输出7,259 Token。详情和所有检索失败见[新库对比报告](../reports/RAG_EXPANSION_EVAL_20261021.md)。同名报告禁止覆盖，再运行应使用新的输出名保留首次结果。

已有同名总报告时命令拒绝覆盖；需要复测时明确指定新的`--output artifacts/local-rag/另一份报告.json`，保留首次成绩。每次使用新的时间戳目录；不自动挑选重试中的最高分，也不把不完整的前次运行拼为同一次完整成绩。

后两条命令针对默认20261021报告：`annotate_rag_eval_acceptance.py`将既有范围提示及引用审计联立成严格安全降级口径，不改原始40题/评分，说明写入`acceptance_cross_check`；`verify_rag_eval_artifacts.py`复查160份真实响应字节和原件一致性，输出`.verification.json`。评测部分通过返回非零是有意行为，不应改门槛消除错误码。

每题使用新会话，原ChatService总超时180秒、原模型输出上限650 token等参数保持不变。模型截断或超时可能使该题失败，这属于真实基线结果。实际起止时间记录在报告，文件名日期不代表伪造执行时间。

## 三层口径

专项修复复测使用 `--output artifacts/local-rag/eval_report_20261021_safety_fix.json`。验收交叉核对和证据核验脚本均支持 `--report artifacts/local-rag/eval_report_20261021_safety_fix.json`，不再限于默认旧报告。修复前后生产源码不同属于预期事实，证据核验会如实记录；冻结题目和评分器必须保持不变。修复调用链、测试语义调整见 [专项修复阅读说明](RAG_SAFETY_REPAIR_20261021.md)。

- 检索层：前4条材料。Doc_Hit分母仅36道有明确期望法律的题目；多文档另报任一/全部。Article_Hit必须同时匹配法律全名、当前版本完整字符串、条号（含增补条号），另报任一/全部。
- 材料组装层：只算真实送入模型的完整材料。预算删掉的条文单独体现，不能算检索未召回；即便模型超时，也从实际请求文本核对已送入材料。
- 回答层：原始模型输出与最终用户答案分别评分；保留被拦截的错误原文。无输出不是零错误，空样本比例为`null`，完整执行另设门槛。

引用审计覆盖显式M编号（含USED_IDS声明）、法律名称及条号、材料编号错绑；无法绑定的裸条号记为待核验。自动校验不衡量所有隐含幻觉或法律推理，不把程序附列原文当作模型引用正确。

边界题评分要求回答明确承认特定地域/规则资料不足，并拒绝给出不能核实的结论；通用错误提示、拦截提示、先说缺资料又给具体条件不算正确范围降级。自然语言规则仍有判定边界，不冒充人工法律语义验收。

## 防御样例与真实模型分开

独立题将固定输出传给**现有**`postprocess`和`deadline_grounding_issues`，不替换其实现。期限样例包括30天内、不得超过30日、至少三个月、跨材料错绑和合法对照；报告实际漏拦与误拦。另有显式假法条、假编号及合法引用对照。

这14题明确标记`deterministic_injection_no_model_call`；没有模型调用，不能混入40道真实咨询题的引用率或宣称是真实模型生成。

## 报告、证据和验收

默认产物：`artifacts/local-rag/eval_report_20261021.json`及同名Markdown《评测与验收结果》。运行证据目录为`artifacts/local-rag/eval-runs/<实际时间戳>/`：

- `metadata.json`：corpus、模型digest、生产及评分源码SHA256、实际参数、初始化网络证据。
- `Q01.json`至`Q40.json`：预期、Top4、实际材料、原始输出、最终输出、错误、阶段耗时、完整请求与响应payload。
- `defense_results.json`：独立注入的材料、预设输出、真实拦截结果。
- `progress.json`：实际已保存题数；`run_failure.json`仅在运行异常时生成，不能代替完整报告。

报告保留所有失败题；不得根据结果修改冻结预期。源码及benchmark运行末尾再次验哈希，防止边跑边改。用户要求的门槛在实施计划中预先写明：完整40题真实推理、Doc>=85%、最终显式错误引用与待核验引用均为0、边界全部正确、非法期限拦截100%、合法对照误拦0。全部通过才合格；不完整为不达标，完整但部分门槛失败为部分通过。Article_Hit完整报告，不私自加用户未规定的分数线。

新增单元测试中的HTTP桩仅验证观测器，不进入真实评测报告。完整单元测试通过也不代表真实问答达标。真实结果请查看运行产物，不在未执行时预填合格数字。

## 本轮实际结果及评分边界

2026-10-21完整运行目录`20261021-122607`：40/40真实推理完成；Doc_Hit@4=36/36，Article任一=34/36、全部=31/36；材料预算后命中相同。原始及最终无材料支持的显式引用各1/40，另1题裸条号待核验；非法期限只拦3/7，合法对照误拦0/4。结论为**部分通过**。

原自动降级规则命中4/4，但其中Q38承认缺德国法资料后仍给出无材料支持的“德国《劳动法》第116条”。因此4/4仅表示范围提示规则命中；联立已有引用审计，严格安全降级为3/4（75%）。JSON的`summary`保留原始自动评分供复现，**正式严格安全验收看`acceptance_cross_check`**；未提高任何分数。

域外法律引用不能盲目映射为国内简称。当前自动审计将Q38的《劳动法》别名识别为国内法律，该标签不证明德国法是否存在相同编号；这里可证实的是引用未被本轮所给中国法材料支持。

首次`20261021-121734`运行14题后因gzip原始payload留证缺陷中止，原件和中止原因保留。修复观测器并通过真实gzip冒烟后完整从头重跑；证据核验确认两次题目、生产代码及评分代码相同，仅观测留证逻辑改变。最近已执行的完整回归222通过（含原有203项）；完整评测后的额外复跑因自动审批额度限制未执行，不将其记为再次通过。


2026-10-21检索复原阶段：当前两路12候选、正文BM25、原RRF直接Top-K；安全与评分源码未改。本阶段仅运行历史候选/历史模型输出的确定性回放，不产生新的40题真实推理成绩；各历史报告仍对应各自代码版本。见[本阶段证据](../reports/RETRIEVAL_ROLLBACK_JUDICIAL_CATALOG_20261021.md)。


## 司法解释收官语料的准入与本次运行

本次入口仍是`python scripts/run_rag_eval.py --output artifacts/local-rag/eval_report_20261022_judicial_closeout.json`。冻结40题、预期、`local_rag/evaluation.py`及生产检索/安全链不变。初始化新增`eval_judicial_evidence.normalize_completion`只读适配：检查第三批COMPLETE、当前corpus/覆盖量、精确点数、逐点核验数、前后真实JUnit与三份原件SHA，再投影为原扩充凭据接口；随后仍须真实检查模型指纹、全部缓存和Qdrant计数。不能绕过原准入检查或把目标数当作实测数。该适配器也纳入运行源码SHA，运行途中不得修改。

真实逐题证据在`artifacts/local-rag/eval-runs/`，每题新会话，记录查询embedding、Qdrant原始payload、聊天流原字节、原始/最终回答、耗时和真实Token。40题完整后用`scripts/verify_rag_eval_artifacts.py --report artifacts/local-rag/eval_report_20261022_judicial_closeout.json`复查，再用`scripts/report_judicial_closeout_eval.py`生成独立综合对比。该脚本只读既有实测报告，不重新评分，不改写原始成绩或失败，不重跑挑分。

案由阅读包含Q03/Q05/Q07/Q08/Q09/Q11全部固定相关案例。须区分召回是否命中、材料组装是否保留、原始模型引用风险与最终拦截；最终显式错误引用为零不代表没有语义问题或已经给出可用建议。历史安全逻辑与检索实验有差异，比较说明可比条件和限制，不把增减全归因于新增司法解释。
