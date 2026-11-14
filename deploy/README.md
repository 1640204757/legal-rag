# 本机数据库入口

`qdrant-gateway.conf` 是 nginx 转发配置，不存储向量。现有数据库容器是 `qdrant`，网关容器是 `legal-rag-qdrant-gateway`，共用 `legal-rag-db` 网络。网关监听本机 16333，转发到数据库 6333；不要清空原数据库或删除存储挂载。

两个容器已创建，日常只需在 Docker Desktop 启动它们。Docker 关闭期间入库不可用。详细使用说明见 `docs/guides/QDRANT_KNOWLEDGE_PIPELINE.md`。
