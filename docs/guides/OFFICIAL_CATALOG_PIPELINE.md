# 官方目录批量采集与入库阅读指南

## 2026-10-21接续批次入口

- `local_rag/date_review.py → scripts/review_date_batch_20261021.py`：读取已保存的官方日期证据，将其与正文版本锚点、原文哈希绑定。默认预检，`--apply`保留旧记录后更新；不会自动给其他法律推断日期。
- `scripts/capture_education_batch_20261021.py`：教育部教育法正文与2021年修改决定的预检及保存，`--apply`才发布资料记录。`catalog_ingest.analyze_document`的`moe`分支只接受唯一`TRS_Editor`正文容器及匹配标题。
- `scripts/capture_moe_additional_20261021.py`：学位法与职业教育法，核对明确版本锚点、末条施行日期和已存NPC目录日期。高等教育法与义务教育法未因下载成功就直接放行。
- `local_rag/batch_storage.py`：补证记录可从原始备份恢复；新增资料先在本批artifacts暂存完整，再改名进入`data/library`。已有不同内容一律拒绝覆盖。
- `scripts/index_date_batch_20261021.py`：默认日期补证批次；`--education`选择教育法，`--moe-additional`选择学位法与职业教育法。三批共用`artifacts/local-rag/verified-batch-worker.lock`，不能同时运行其他写缓存入口。固定批次数字用于拒绝范围意外变化，不是未来通用任务入口。旧`catalog_batch_worker.py`仍是旧批次专用。
- `scripts/finalize_resume_batch_20261021.py`：三个批次独立入库证据齐备后，复核五份快照和当前语料指纹，再更新本轮报告与来源v6；不代替实际入库。

依次执行预检、保存、完整回归、入库。每批`artifacts/local-rag/<批次>/qdrant_verification.json`存在且精确点数一致，才报告数据库写入完成。新旧corpus保留；新增资料数量不等于集合历史总点数。

教育法正文末条仍写原始1995年施行日；本批修订版施行日2021-04-30取自另存的官方修改决定，不用网页发布时间替代。监察法同样区分原法施行条款与2024年修正决定的2025-06-01施行日。

## 从哪里看

先看 `local_rag/catalog_ingest.py`：目录发现、原文下载、条文解析、异常隔离均在此模块。它不生成向量。
再看 `local_rag/ingest.py` 的 `load_extra()`：只加载 `PARSED` 记录，核对原文与条文文件哈希。
最后看 `local_rag/retrieval.py` 的 `Retriever.build()` 和 `local_rag/qdrant.py`：缓存复用、生成向量、按批次写入数据库并核对数量。

## 运行顺序

在项目根目录，使用已有本机解释器并配置根目录及 `.local-deps` 到 PYTHONPATH：

```text
python -m local_rag.catalog_ingest discover
python -m local_rag.catalog_ingest capture
python -m local_rag.index
```

`discover` 保存市场监管总局法律目录和司法部行政法规目录的全部公开分页、哈希及总数对照。目录齐全不代表正文齐全，也不等于全国所有法律资料。
`capture --limit 5` 可限制新增正文请求数；已保存记录复用，不自动重新抓取失败记录，不自动更新已存在同名法律。
`capture` 长批次中每完成一份原子保存记录；中断后重跑可继续。超时资料保留失败记录；重试应另存旧记录和新采集时间，不能覆盖证据后冒充原采集。
`index` 才会实际生成向量和写入 Qdrant；`load_corpus()` 返回的“已加载”是磁盘资料数量，入库成功以数据库 `ready`、当前 `corpus_id` 的精确点数为准。

## 数据位置与状态

- `data/catalogs/official-20261020/<provider>/`：原始目录页及 `catalog.json`。
- 同目录 `collection_report.json`：逐条状态与错误；并非成功入库清单。
- `data/library/catalog-*/source.html`：保留下载字节；不得为了排版修改。
- 同目录 `articles.json`：按条切分正文；`record.json` 记录日期、来源、原文及产物哈希。
- 国家法律法规数据库公开目录如有保存，单列 `npc/`；只有元数据的记录不得算作正文或向量覆盖。

`PARSED` 仅表示标题、正文容器、连续条号和末条施行文字等结构检查通过，并且取得的施行日期不晚于采集当天。它不等于专家确认现行有效，更不等于个案适用正确。
`FUTURE_EFFECTIVE` 保存但不入当前库；`DATE_REVIEW_REQUIRED` 日期待核对，保存但不入库；`QUARANTINED` 解析/边界异常；`UNAVAILABLE` 请求未取得正文。
`EXISTING_TITLE_VERSION_REVIEW_REQUIRED` 表示已有同名资料，本轮未静默替换；必须后续比对版本证据，不能因同名就认定最新版已覆盖。
`OUTSIDE_NATIONAL_CATALOG_REVIEW` 是未列入该期全国法律对照目录的条目，需要另行分类审查，不据此宣称废止。

## 版本与完整性边界

目录可能展示“最新公布但尚未施行”的版本。当前实现隔离未来版本，不自动用其替换仍适用的旧版；因此这类法律可能仍存在当前正文缺口。
市场监管总局网页日期不直接当成法律修订日期；无法确认修订日期时，用快照哈希作版本身份。
连续条号及末条文字只能发现部分截断，不能独立证明附件、表格、全部款项及修订整合均完整。原文快照一致也不等于与实时官网一致。
不同来源目录范围不同；全国法律目录、行政法规目录、司法解释、地方性法规、案例必须分别计数。公开目录总数变化应标记新批次，不硬写固定全量。

## 检查与恢复

```text
python -m pytest tests_local prototype/tests -q --rootdir .
python -m local_rag.index --search "企业破产申请由谁提出"
```

测试中的官方页面是最小结构样例；通过测试不代表全部真实网页均正确，真实批次另作哈希/条号/状态检查。
Qdrant 按当前语料指纹隔离新旧批次；重复入库不增加当前批次点数，不删除其他项目集合。数据库集合总点数包括历史批次，不能当成当前覆盖量。
新增批次完成记录见 `docs/reports/OFFICIAL_CATALOG_BATCH_20261021.md`；若该报告尚未生成，说明批次尚未完成，不能先宣称已入库。

## 批量任务进度与恢复

本轮收尾脚本为 scripts/catalog_batch_worker.py：等待已启动的采集完成，普通网络错误只重试一次；访问限制立即失败退出。通过原文哈希/重解析和完整测试后，再调用本机模型与Qdrant。最终检查重复同步和固定查询，自动生成批次报告。

进度文件：rtifacts/local-rag/catalog-batch/status.json。WAITING_FOR_CAPTURE 是等待正文采集，EMBEDDING_OR_SYNC 的 completed_vectors 是缓存完成数量，不能当成数据库点数。只有 COMPLETE 且 qdrant_verification.json 存在、批次计数一致，才表示本批入库完成；仍不表示全国全量。FAILED 需先查原因，不应忽略错误继续。

任务依赖本机保持运行及Docker/Ollama可用；没有安排定时任务或云端作业。断电后进程不会自行启动，重新运行脚本会复用已采资料和已保存向量；若有残留 worker.lock，必须先确认原进程不存在再清理，不能同时运行两份写缓存。
