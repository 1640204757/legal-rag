# 开源准备：使用边界、资料来源与发布检查

2026-11-03起按下一阶段方案执行。代码起点为 `b966474`。本记录区分已实现、已验证及待处理事项，不作为法律现行性或完整安全认证。

## 本轮工程整理

### 发布质量要求与对外说明

用户要求明确：正式开源前尽量排除已知工程问题，将常见故障和处理办法写清楚，同时不承诺所有问题都能处理、所有结果都正确或所有环境都兼容。发布前需修复已知严重工程缺陷；支持范围内可稳定复现的故障应解决或明确列为发布阻断项，不能只补免责声明后继续称“无问题”。

[使用手册](USER_GUIDE.md)已补充使用范围及常见问答：页面与后端分别就绪、候选出现但未送模、未检索到不等于无依据、材料阅读为何不直接判责、重复回答差异、资源不足、本机访问/联网边界，以及匿名化反馈要求。正式发布还应列明已知缺口、复现/规避方式、验证环境与反馈入口；尚无专用漏洞私密反馈渠道和最终发布包，不能写成已具备。

本次补充仅修改文档，核对本地链接及差异格式，不重跑模型、检索测量或单元测试；下方1049/1046项为上一轮工程验证结果。

### 已实现：安全启动、只读自检与使用文档

新增 [使用手册](USER_GUIDE.md)，按“资料与软件准备→Python→模型→数据库→自检→启动→阅读结果→更新/备份→排错”说明，不让新用户复制作者本机配置。明确正式开源发布与法律判断可靠性是不同验收目标，当前尚未发布一个已验收的完整安装包。

原启动器按进程名结束Ollama、根据PID文件/端口结束Web；模型适配器也会因缺模型标签结束已有Ollama。它们可能影响其他项目，且旧PID可能已被别的进程使用。现在两处都不再按名称杀服务：

- `run_local.py` 只复用带同目录身份的Web；其他监听者、旧版缺身份或暂未响应均保留并报错。只停止本次Popen持有的Web子进程，PID文件仅供诊断。
- `--check`只读核对选定的64位CPython 3.12、依赖、资料原件、模型标签、数据库健康及端口归属；不启动服务、不建索引、不调用4B、不输出访问令牌，也不跟随本机端点重定向。
- `local_runtime.py`让Web和启动器共享目录摘要，身份字段不直接暴露目录、不是认证或终止进程凭据；原Token/Origin边界不变。
- 启动器和模型适配器统一显式`OLLAMA_MODELS`优先；已有服务目录不能由客户端改变，不自动迁移模型。适配器只在默认本机端口空闲时尝试启动，缺模型只报就绪状态。
- 独立安装暴露3项旧测试依赖未跟踪私人报告的问题。增加3项公开合成引用回归；历史报告存在时仍核对真实记录，缺失时明确跳过，不把私人原始回答带进发布包。

### 已验证：当前目录与独立环境

- 当前指定解释器：**1049通过、14跳过、35子测试通过**，退出0，52.75秒；仍有1条既有Starlette/httpx弃用警告。新回归覆盖保留已有Ollama、拒绝接管另一目录Web、陈旧PID、只读自检不打印令牌、只停止自己的句柄、显式模型目录、错误本机地址、异常HTTP身份与解释器配置。
- 在D盘项目内新建独立代码副本和隔离虚拟环境，从官方PyPI安装原`requirements-local.txt`成功；没有复制`.local-deps`、作者Python路径、模型或向量缓存。首次全量因私人报告缺失3项失败，按上面方式修正后：**1046通过、17跳过、35子测试通过**，退出0，48.76秒。多出的3个跳过正是未提供私人历史报告，不是隐藏断言失败。
- 独立副本自检的Python/依赖/语料全部通过，实际加载878份规范、6份案例摘要、46,701段。它与主目录共用本机模型和数据库；18765属另一目录，因此按设计返回`occupied_or_unverified`、退出2，没有接管或结束服务。
- 主目录旧Web缺少身份字段时，自检退出2且原PID46564保持运行。确认其解释器/命令及语料后由维护操作单独重启，新启动器成功启动Web PID43808，再次启动复用同一PID；主目录自检退出0。网页静态资源正常，两个模型就绪、`database.ready=true`、`index.state=ready`，语料指纹与46,701段均不变，Ollama11434和Qdrant6333/16333监听进程与重启前一致。
- 本轮不改`Retriever.search()`、词表、排序、分数、阈值或送模条数，未额外调用真实4B。此前检索测量dev51/61、holdout13/16、v2 36/36及材料阅读的真实生成/离线重放证据仍是各自轮次结果，不冒称本轮重跑成绩。

证据保存在本机忽略目录`artifacts/local-rag/release-readiness/`：`tests-final.log`、`web-check.json`、`check-before-restart.json`、独立环境`install.log/check.json/tests-final.log`。只保存服务状态白名单，不保存访问令牌或holdout逐题分析。该独立目录副本包含本地资料，**是私有验证副本，不是获准再分发的数据包**。

### 待完成：正式公开前的实质门槛

1. **冻结资料发布清单和兼容数据包**。区分法规原文、网页附带脚本/编辑内容、案例摘要、目录证据和历史日志。[著作权法第五条](https://www.nsfc.gov.cn/p1/2871/2872/69475.html)列明其不适用的官方文件范围；这些原文本身与网站其他内容的权利不同，不能笼统套MIT，也不能笼统说所有法条都需另获版权许可。当前记录中的本地使用标记尚未完成复核；加载器依赖原件/哈希，不能直接删原件只上传解析JSON。
2. **完整发布内容与隐私复核**。当前树规则扫描尚未覆盖Git历史及所有非UTF-8/二进制文件。19个保留历史记录含机器路径；如何保留历史与准备公开包需在具体清单上决定，未擅自删除证据或改写历史。私有回答、访问令牌、缓存、运行日志、作者配置和未公开盲测均不入公开包。
3. **用最终发布包完成另一台机器的安装验收**。本轮是同机独立目录/虚拟环境，使用既有模型和Qdrant；还没证明全新Windows、全新Docker存储、模型下载、从零完整入库及整个安装文档在外部机器无遗漏。Docker卷也不能仅凭项目在D盘就认定位于D盘。
4. **形成可维护的发布记录**。最终源码/数据版本、校验和、兼容环境、已知缺口、贡献与漏洞反馈方式应随版本确定；当前没有远端发布、自动CI或新的人工盲测成绩。不把仍在调试的已用题集宣称全新盲测。

### 数据打包的具体边界

- 可随代码考虑公开：程序、原创文档、许可、无隐私的合成测试及来源清单；现有评测题若公开用于复现，就不再宣称其盲测性。
- 单独筛选、核验版本和来源后再决定：`data/library`、`data/raw`、`data/normalized`、`data/cases`及必要的目录证据。原始快照和解析产物应保持可核验关系，未发布目录不能写成已下载可用。
- 不随源码公开：`.local-python-path.txt`、`.env`、`.local-deps/.venv`、`data/runtime`、`data/local_index`、`data/ollama_models`、数据库卷、`artifacts`原始回答和评测明细、用户私有题集。模型和依赖由使用者按各自许可取得。
- 当前Git已跟踪的资料不会因为补了`.gitignore`就从历史消失。真正发布必须核对最终清单及Git历史，不能把“目录被忽略”当作过去从未提交过的证明。

因此，项目可以继续向正式开源版本整理，但当前不能写成“已经完全完善、直接整目录上传即可”。剩余重点是可公开资料与可复现交付，不是继续为评测题微调检索。

## 已实现：使用声明与代码许可

README顶部和网页问答区顶部常驻中文使用声明。网页使用静态文本节点，不通过 `innerHTML` 注入；开始会话后不会随引导卡隐藏。未改模型提示词、送模数量或检索参数。

用户选择MIT后添加根目录 `LICENSE`，权利人署名为 OpenLawGuide contributors。代码和项目原创文档适用MIT；来源数据、模型与第三方依赖的权利应分别核对。标准许可文本见 [OSI MIT](https://opensource.org/license/mit)。

## 已盘点：数据字段与来源

- `data/library/*/record.json`：876份已解析记录均有 `version`，860份有 `effective_date`，4份有 `version_review_status`，692份有 `version_applicability`；全部 `validity_status=UNVERIFIED`。`version` 是采集的版本标记，不是每轮在线验证日期。874份 `reuse_permission=UNKNOWN_LOCAL_RETRIEVAL_ONLY_NOT_TRAINING_OR_REDISTRIBUTION`，另2份缺该字段；尚未完成数据再分发许可复核。
- `data/normalized/*.json`：2份有 `version_id/publish_date/effective_date`，保存 `validity_status=IN_FORCE`；`citation_eligibility` 另列原文核验、版本确认、资格和事实匹配状态。不能把字段中的 `IN_FORCE` 当作本次在线核验结论。
- `data/cases/*.json`：6份劳动争议典型案例摘要，`publish_date`齐全；来源为最高人民法院发布稿，保留 `source_url` 与引用定位，仅供参考，不是生效判决书全文。
- 来源定位入口：`data/raw/capture_manifest.json`、上述记录的 `url/source_url`、`local_rag/ingest.py` 的 `SOURCES`、国家库目录与采集批次。官网域名包括国家法律法规数据库、最高人民法院、国务院部门等；以每份记录为准，不用统一域名替代原链接。

`load_corpus()` 对既有规范逐字比对快照、对案例比对发布稿；新增资料由 `load_extra()` 验证原件/解析产物哈希与条号。它们能检出与已存快照不一致，不能证明采集源从未出错、现行版本完整、个案推理正确或资料已获再分发许可。原始快照本轮未修改。

## 发布检查记录

当前工作树只读扫描的本地明细保存在项目内 `artifacts/local-rag/open-source-readiness/audit.json`；报告只列相对文件路径、行号和风险类别，不复制疑似秘密或holdout逐题内容。仅扫描当前已跟踪文件，未覆盖Git历史或所有秘密格式；详细结果和验证结论随本轮补充。

## 后续步骤的边界

- **后续已接入材料阅读许可**：不再让默认问答自由给个案处置结论，完整来源节选与固定通用步骤之外的内容整段拒绝。金额/倍数等旧检查及已有前端改动保留。最终全量1028通过、14跳过、35子测试通过；真实4B同批20题完成20/20，再对纯输出边界修正做20题同输入离线重放，17题放行、3题拒绝，不能把重放计作新推理或宣称消除法律适用风险。细节与剩余格式/摘录缺口见 [材料阅读许可](MATERIAL_READING.md)。
- 本轮Web已加载新许可，索引/数据库ready，原集合46,701段。单次“直接认定我是否违法、应赔多少”的接口验证返回通用材料阅读说明，无直接个案结论；Qdrant/网关/Ollama健康200。检索dev51/61、holdout13/16、v2 36/36，快照零变化、MRR不变。没有把单次接口或语法许可视为完整法律适用审查。
- 新 `blind_v4` 题目等待用户或他人提供，AI不代出盲题。AI草拟标注须人工确认后才冻结、跑基线；在此之前不声称具有新盲测成绩。现有holdout反复作为选型门槛，已不等同于从未使用的盲测。
- 候选池20/30/40先复用同批真实语义结果比较；只允许改变每路池深，其他规则不顺手调整。任何既有前4/前8命中退步或holdout MRR下降均不采用。
- **该比较已完成，30/40均未采用**：出现dev、holdout、v2既有命中退步，生产保留20，完整过程见 [检索实施记录](RETRIEVAL_MEASUREMENT.md#实施记录)。没有新盲测成绩，也未据此调整其他参数补救扩池。
- 保留20的正式测量与验收均退出0：dev前4/前8为51/61，holdout13/16，v2 36/36，三组快照零变化；holdout MRR保持0.208333。最终测量目录 `20261103-163025-980858`，检索验收目录 `20261103-163104-620992`，均在项目内 `artifacts/local-rag` 的对应运行目录中。
- 更大向量模型下载已由用户决定留到下一轮；本轮未下载、未切默认或覆盖旧缓存/集合。
- 后续用户单独授权的送模4/5/6条预算试验已完成，正确条文送模仅86→87→86/156，用户确认按明显收益门槛保留4条；生产/展示请求和预算均未改，未跑候选真实回答对照。新增预算尺子及参数透传回归，当前版本另建立新的20题4B回答基线。过程、数据与未验证项见 [送模试验记录](PROMPT_MATERIAL_EXPERIMENT.md)。
- 本轮新基线已完成20/20，实际送模含正确条文16/20；仍有2题拦截、2题引用审计缺陷及1题原始声明漏列（已修复但不抹去缺陷），因此不能称生成质量或个案适用已全面验收。候选扩量未采用，也无候选质量对比成绩。
- 送模试验工具与观测修复后最新全量测试992通过、14跳过、35子测试通过，1条既有弃用警告。生产 `_run()`与检索器源码没有改动；展示 `[4, 8]` 回归断言保留并通过。
- 本轮全量检索测量与验收均通过：三组前4/前8快照零变化、MRR不变，dev51/61、holdout13/16、v2 36/36，展示仍73/26/36；未向Git添加原始回答证据或holdout报告。
- 已实现并完成本地全量测试：每张依据卡片写明采集版本和未在线复核现行性；赔偿倍数与确定金额对不上所引材料时拦截。全量测试 1000 通过、14 跳过、35 子测试通过。未改检索排序，也未新出盲题，未重跑真实 4B 回答。法规原文再分发许可仍未确认，不能把资料库视为已可随 MIT 公开分发。

## 已验证：当前工作树泄露检查与页面声明

- 盘点4886个已跟踪文件，其中381个二进制或非UTF-8文件未作文本规则扫描；`.local-python-path.txt`、`.env` 均未被跟踪，`.gitignore` 已忽略它们，以及 `artifacts/`、`data/runtime/`、模型和本地向量缓存。
- 已跟踪 `artifacts/` 文件数为0，因此该目录中没有待发布的holdout逐题报告。评测题集本身仍在 `data/eval`，现有holdout不应宣称为从未接触的盲测。
- 私钥头、常见供应商令牌、JWT形态及带引号凭据赋值扫描均未命中。此为规则扫描结果，不代表不存在任何秘密；未扫描Git历史。
- 首次扫描有32个文件含Windows绝对路径。用户允许改写文档和旧测试路径、保留历史日志后，13个文档/测试入口文件已改为相对路径或自动定位根目录；复扫剩19个历史记录文件，未删除或取消跟踪。文档链接目标保持存在；历史报告中的原路径只在文档摘录中转为相对展示，原始证据不动。
- 建议继续忽略未来生成的 `data/catalogs/**/traceback.txt`、`tests*.txt` 和测试临时目录；历史已跟踪记录不会因添加忽略规则而自动消失，应在确认后单独处理。不能一概忽略 `data/catalogs/`，其中还有来源目录及证据文件。
- 路径修改后项目解释器全量测试：986通过、14跳过、35子测试通过，1条既有Starlette/httpx弃用警告，退出0，39.65秒。未增加与静态文案实现重复的测试。兼容旧入口 `scripts/run_tests.py` 改用文件位置定位项目根目录；推荐入口仍为 `run_workspace_tests.py`。
- 旧入口另外从项目内其他工作目录执行仅收集模式，1000项收集成功、退出0，确认不再依赖作者本机绝对路径；此检查不代替上述完整执行。日志为 `legacy-runner-collect.log`。
- 本机Web返回200，声明位于可折叠引导卡之外，数据库ready=true、索引ready；检查未保存令牌。未进行窄屏浏览器视觉验收。证据：`artifacts/local-rag/open-source-readiness/web-check.json` 与 `tests.log`。

### 首次扫描路径清单（文档/旧测试已处理，历史记录保留）

- `data/catalogs/coverage-audit-20261022/administrative_reconciliation.json`：行 7, 17
- `data/catalogs/judicial-core-20261021/batch-001/approval-verifier-red.txt`：行 2, 4
- `data/catalogs/judicial-core-20261021/batch-001/no-overwrite-red.txt`：行 2, 4, 7, 10, 13
- `data/catalogs/judicial-core-20261021/batch-001/tests-final.txt`：行 8
- `data/catalogs/judicial-core-20261021/batch-001/tests.txt`：行 8
- `data/catalogs/judicial-core-20261021/local-batch-001/attempts/20261022-040237-88941433/traceback.txt`：行 2, 4, 7, 10, 13
- `data/catalogs/judicial-core-20261021/local-batch-001/attempts/20261022-044512-3508c90b/tests-before-publish.txt`：行 8
- `data/catalogs/judicial-core-20261021/local-batch-001/attempts/20261022-044512-3508c90b/traceback.txt`：行 2, 4, 7, 10, 13
- `data/catalogs/judicial-core-20261021/local-batch-001/attempts/20261022-045826-34e757d6/tests-before-publish.txt`：行 8
- `data/catalogs/judicial-core-20261021/local-batch-001/attempts/20261022-045826-34e757d6/tests-final.txt`：行 8
- `data/catalogs/judicial-core-20261021/local-batch-002/attempts/20261022-051219-938d9211/traceback.txt`：行 2, 4, 7, 10, 13, 16, 19, 22, 24
- `data/catalogs/judicial-core-20261021/local-batch-002/attempts/20261022-052006-49f947a5/tests-before-publish.txt`：行 8
- `data/catalogs/judicial-core-20261021/local-batch-002/attempts/20261022-052006-49f947a5/tests-final.txt`：行 4, 9
- `data/catalogs/judicial-core-20261021/local-batch-002/attempts/20261022-052006-49f947a5/traceback.txt`：行 2, 5, 7, 12, 15, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 47, 49, 52
- `data/catalogs/judicial-core-20261021/local-batch-002/attempts/20261022-052452-26b4cd32/tests-before-publish.txt`：行 8
- `data/catalogs/judicial-core-20261021/local-batch-002/attempts/20261022-052452-26b4cd32/tests-final.txt`：行 8
- `data/catalogs/judicial-core-20261021/local-batch-003/attempts/20261022-054132-8c5fcb89/tests-before-publish.txt`：行 8
- `data/catalogs/judicial-core-20261021/local-batch-003/attempts/20261022-054132-8c5fcb89/tests-final.txt`：行 8
- `data/catalogs/national-laws-20260828/coverage.json`：行 48, 70, 86, 108, 520, 530, 942, 976, 1406, 1416, 1600, 1766
- `docs/guides/PYCHARM_PROJECT_READING.md`：行 7, 280
- `docs/guides/QDRANT_KNOWLEDGE_PIPELINE.md`：行 7
- `docs/guides/REAL_RAG_EVALUATION.md`：行 18
- `docs/reports/RAG_JUDICIAL_CLOSEOUT_EVAL_20261022.md`：行 60, 321
- `docs/requirements/legal_assistant_prd.md`：行 5
- `docs/requirements/legal_case_rag_requirements_review.md`：行 5
- `prototype/tests/test_citation_binding.py`：行 8
- `prototype/tests/test_citation_states.py`：行 9
- `prototype/tests/test_import_and_search.py`：行 4
- `prototype/tests/test_no_material_and_postprocess.py`：行 4
- `prototype/tests/test_rag_package.py`：行 4
- `prototype/tests/test_verbatim_and_cases.py`：行 4
- `scripts/run_tests.py`：行 5
