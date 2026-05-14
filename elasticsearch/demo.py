"""
Elasticsearch 前置知识学习模块 - CRUD Demo

独立可运行脚本，覆盖 Elasticsearch 在 SmartRecruit 项目中使用的核心操作。
连接参数与项目一致：http://localhost:9200，索引名使用 prerequisite_demo_chunks 避免污染项目数据。
"""

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk


# ---------------------------------------------------------------------------
# 1. connect - 连接 Elasticsearch 实例
# ---------------------------------------------------------------------------
def connect(host="http://localhost:9200"):
    """
    连接 Elasticsearch 实例并验证连通性。

    Args:
        host (str): ES 实例地址，默认 http://localhost:9200

    Returns:
        Elasticsearch: 已连接的 ES 客户端实例

    Raises:
        ConnectionError: 无法连接到 ES 实例时抛出
    """
    # 创建 Elasticsearch 客户端，指定主机地址
    # 项目中对应：self.es_client = Elasticsearch(config.ES_HOST)  (步骤 2.15)

    # ping() 验证连接


    es = Elasticsearch(host)  # ← 应该使用传入的 host 参数

    if es.ping():
        info = es.info()
        print(f"集群: {info['cluster_name']}, 版本: {info['version']['number']}")

    # ping() 方法验证连接是否成功，返回 True/False
    if es.ping():
        info = es.info()
        print(f"[connect] 连接成功！集群名称: {info['cluster_name']}, 版本: {info['version']['number']}")
    else:
        raise ConnectionError("无法连接到 Elasticsearch，请确认服务已启动")
    return es


# ---------------------------------------------------------------------------
# 2. create_index - 创建索引（含 mapping 定义）
# ---------------------------------------------------------------------------
def create_index(es, index_name="prerequisite_demo_chunks"):
    """
    创建索引并定义 mapping，模拟 SmartRecruit 中简历子块的存储结构。

    Mapping 包含：
    - content: text 类型，用于全文搜索（BM25）
    - metadata: object 类型，存储文档元信息（id、hash、parent_content 等）

    Args:
        es (Elasticsearch): ES 客户端
        index_name (str): 索引名称

    Returns:
        dict: 创建结果
    """
    # 如果索引已存在，先删除（demo 场景，保证幂等）



    # 定义索引的 mapping（字段映射）
    # 项目中对应：步骤 2.16 检查索引是否存在并创建（项目未显式定义 mapping，使用动态映射）
    # 这里显式定义以演示最佳实践
    mapping = {
        "mappings": {
            "properties": {
                # content 字段：text 类型，启用全文检索
                # ES 会自动建立倒排索引，支持 match/term 等查询
                "content": {
                    "type": "text",
                    # 使用标准分词器（默认），适用于英文；中文场景建议用 ik_max_word
                    "analyzer": "standard",
                },
                # metadata 字段：object 类型（默认），可以包含任意子字段
                # 项目中存储：id、hash、parent_content 等
                "metadata": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "keyword"},          # 精确匹配，不分词
                        "hash": {"type": "keyword"},        # 文档哈希值
                        "parent_content": {"type": "text"}, # 父块内容
                    },
                },
            }
        }
    }

    # 创建索引
    result=es.indices.create(index="prerequisite_demo_chunks", body=mapping)

    print(f"[create_index] 索引创建成功: {index_name}")
    print(f"  确认信息: {result['acknowledged']}")
    return result


# ---------------------------------------------------------------------------
# 3. index_document - 索引单条文档
# ---------------------------------------------------------------------------
def index_document(es, index_name="prerequisite_demo_chunks"):
    """
    索引单条文档（指定文档 ID）。

    模拟 SmartRecruit 中将简历子块写入 ES 的操作。
    项目中对应：步骤 4.15
        self.es_client.index(index=..., id=chunk.metadata["id"], document=es_doc)

    Args:
        es (Elasticsearch): ES 客户端
        index_name (str): 索引名称

    Returns:
        dict: 索引结果
    """
    # 构造文档内容，模拟简历子块
    doc = {
        "content": "张三，5年Java开发经验，熟悉Spring Boot、MySQL、Redis，有大型分布式系统设计经验",
        "metadata": {
            "id": "chunk_001",
            "hash": "abc123def456",
            "parent_content": "张三的完整简历内容...",
        },
    }

    # 使用 index() 方法写入文档
    # 参数说明：
    #   index  - 目标索引名
    #   id     - 文档唯一标识（项目中使用 chunk.metadata["id"]）
    #   document - 文档内容（dict）
    result=es.index(index="prerequisite_demo_chunks", id="chunk_001", document=doc)
    print(f"[index_document] 文档已索引，ID: {result['_id']}, 结果: {result['result']}")
    return result


# ---------------------------------------------------------------------------
# 4. bulk_index - 批量索引文档
# ---------------------------------------------------------------------------
def bulk_index(es, index_name="prerequisite_demo_chunks"):
    """
    批量索引多条文档，演示 bulk API 的使用。

    虽然 SmartRecruit 当前使用循环逐条 index()（步骤 4.15），
    但 bulk 是 ES 推荐的批量写入方式，性能更好（减少网络往返）。

    Args:
        es (Elasticsearch): ES 客户端
        index_name (str): 索引名称

    Returns:
        tuple: (success_count, errors_list)
    """
    # 准备批量写入的文档列表
    # 每个文档是一个 dict，包含 _index、_id 和 _source
    docs = [
        {
            "_index": index_name,
            "_id": "chunk_002",
            "_source": {
                "content": "李四，3年Python开发经验，熟练使用Django、Flask，有机器学习项目经验",
                "metadata": {
                    "id": "chunk_002",
                    "hash": "def789ghi012",
                    "parent_content": "李四的完整简历内容...",
                },
            },
        },
        {
            "_index": index_name,
            "_id": "chunk_003",
            "_source": {
                "content": "王五，8年前端开发经验，精通React、Vue、TypeScript，有性能优化和团队管理经验",
                "metadata": {
                    "id": "chunk_003",
                    "hash": "ghi345jkl678",
                    "parent_content": "王五的完整简历内容...",
                },
            },
        },
        {
            "_index": index_name,
            "_id": "chunk_004",
            "_source": {
                "content": "赵六，2年Go开发经验，熟悉微服务架构、Docker、Kubernetes，有云原生项目经验",
                "metadata": {
                    "id": "chunk_004",
                    "hash": "jkl901mno234",
                    "parent_content": "赵六的完整简历内容...",
                },
            },
        },
    ]

    # 使用 bulk() 批量写入
    # bulk() 接受一个可迭代对象，每个元素是一个操作描述
    # 返回 (success_count, error_list)
    success, errors = bulk(es, docs)
    print(f"[bulk_index] 批量索引完成，成功: {success} 条")
    if errors:
        print(f"  错误信息: {errors}")
    return success, errors


# ---------------------------------------------------------------------------
# 5. get_document - 按 ID 获取文档
# ---------------------------------------------------------------------------
def get_document(es, index_name="prerequisite_demo_chunks"):
    """
    按文档 ID 获取完整文档内容。

    项目中对应：步骤 7.21 中通过 hit['_id'] 获取 ES 结果中的文档 ID
    也用于调试和验证文档是否正确写入。

    Args:
        es (Elasticsearch): ES 客户端
        index_name (str): 索引名称

    Returns:
        dict: 文档内容（包含 _source）
    """
    doc_id = "chunk_001"

    # get() 方法通过 ID 获取单个文档
    # 返回 dict，包含 _index、_id、_version、_source 等字段
    result = es.get(index="prerequisite_demo_chunks", id="chunk_001")

    print(f"[get_document] 文档 ID: {result['_id']}")
    print(f"  内容: {result['_source']['content'][:50]}...")
    print(f"  元数据: {result['_source']['metadata']}")
    return result


# ---------------------------------------------------------------------------
# 6. search_match - match 全文搜索
# ---------------------------------------------------------------------------
def search_match(es, index_name="prerequisite_demo_chunks"):
    """
    使用 match 查询进行全文搜索。

    这是 SmartRecruit 中 ES 的核心用法。
    项目中对应：步骤 7.18
        self.es_client.search(index=..., body={"query": {"match": {"content": query}}, "size": k})

    match 查询会对搜索词分词后在倒排索引中查找匹配的文档，
    结果按 BM25 评分排序。

    Args:
        es (Elasticsearch): ES 客户端
        index_name (str): 索引名称

    Returns:
        list: 搜索结果列表
    """
    query = "Java开发经验"

    # 构造 match 查询
    # match 会对 "Java开发经验" 分词，然后在 content 字段的倒排索引中查找
    body = {
        "query": {"match": {"content": "Java开发经验"}},
        "size": 5,
    }

    # 执行搜索
    result = es.search(index="prerequisite_demo_chunks", body=body)

    hits = result["hits"]["hits"]

    print(f'[search_match] 查询: "{query}"，命中 {len(hits)} 条')
    for hit in hits:
        score = hit["_score"]
        content = hit["_source"]["content"][:40]
        print(f"  ID: {hit['_id']}, BM25 评分: {score:.4f}, 内容: {content}...")
    return hits


# ---------------------------------------------------------------------------
# 7. search_bool - bool 组合查询（must + filter）
# ---------------------------------------------------------------------------
def search_bool(es, index_name="prerequisite_demo_chunks"):
    """
    使用 bool 查询组合多个条件：must（必须匹配）+ filter（过滤，不影响评分）。

    虽然 SmartRecruit 当前仅使用简单的 match 查询，
    但 bool 查询是 ES 中最常用的查询类型，掌握它有助于理解复杂检索场景。

    bool 查询的四种子句：
    - must:    必须匹配，参与评分
    - should:  至少匹配一个（可配置），参与评分
    - must_not: 必须不匹配，不参与评分
    - filter:  必须匹配，不参与评分（性能更好，ES 会缓存结果）

    Args:
        es (Elasticsearch): ES 客户端
        index_name (str): 索引名称

    Returns:
        list: 搜索结果列表
    """
    body = {
        "query": {
            "bool": {
                # must: 全文匹配"开发经验"，参与 BM25 评分
                "must": [
                    {"match": {"content": "开发经验"}},
                ],
                # filter: 精确过滤 metadata.id 以 "chunk_00" 开头
                # filter 不参与评分，ES 会利用缓存加速
                "filter": [
                    {"prefix": {"metadata.id": "chunk_00"}},
                ],
            }
        },
        "size": 10,
    }

    result = es.search(index=index_name, body=body)
    hits = result["hits"]["hits"]

    print(f"[search_bool] bool 查询命中 {len(hits)} 条")
    for hit in hits:
        score = hit["_score"]
        content = hit["_source"]["content"][:40]
        print(f"  ID: {hit['_id']}, 评分: {score:.4f}, 内容: {content}...")
    return hits


# ---------------------------------------------------------------------------
# 8. update_document - 部分更新文档字段
# ---------------------------------------------------------------------------
def update_document(es, index_name="prerequisite_demo_chunks"):
    """
    部分更新文档字段（无需重写整个文档）。

    使用 update API 的 doc 参数，只更新指定字段，其他字段保持不变。
    这在需要修改文档的某个属性时非常有用，比如更新简历的评分状态。

    Args:
        es (Elasticsearch): ES 客户端
        index_name (str): 索引名称

    Returns:
        dict: 更新结果
    """
    doc_id = "chunk_001"

    # 使用 update() 方法部分更新
    # doc 参数指定要更新的字段及其新值
    result=es.update(
        index="prerequisite_demo_chunks",
        id="chunk_001",
        body={"doc": {"content": "更新后的内容..."}},
    )
    print(f"[update_document] 文档 {doc_id} 已更新，结果: {result['result']}")

    # 验证更新后的内容
    updated = es.get(index=index_name, id=doc_id)
    print(f"  更新后内容: {updated['_source']['content'][:60]}...")
    return result


# ---------------------------------------------------------------------------
# 9. delete_document - 按 ID 删除文档
# ---------------------------------------------------------------------------
def delete_document(es, index_name="prerequisite_demo_chunks"):
    """
    按文档 ID 删除单个文档。

    项目中暂未使用删除操作（简历存储后不会删除子块），
    但这是 CRUD 中必备的操作。

    Args:
        es (Elasticsearch): ES 客户端
        index_name (str): 索引名称

    Returns:
        dict: 删除结果
    """
    doc_id = "chunk_004"

    # 使用 delete() 方法按 ID 删除文档
    result=es.delete(index="prerequisite_demo_chunks", id="chunk_004")
    print(f"[delete_document] 文档 {doc_id} 已删除，结果: {result['result']}")

    # 验证删除（预期抛出 NotFoundError）
    try:
        es.get(index=index_name, id=doc_id)
    except NotFoundError:
        print(f"  验证: 文档 {doc_id} 已不存在（符合预期）")
    return result


# ---------------------------------------------------------------------------
# 10. delete_index - 删除整个索引
# ---------------------------------------------------------------------------
def delete_index(es, index_name="prerequisite_demo_chunks"):
    """
    删除整个索引及其所有文档。

    危险操作！生产环境中通常不会删除索引。
    这里用于 demo 清理，确保下次运行时环境干净。

    Args:
        es (Elasticsearch): ES 客户端
        index_name (str): 索引名称

    Returns:
        dict: 删除结果
    """
    # 先确认索引存在
    if es.indices.exists(index=index_name):
        result = es.indices.delete(index=index_name)
        print(f"[delete_index] 索引 {index_name} 已删除，确认: {result['acknowledged']}")
    else:
        print(f"[delete_index] 索引 {index_name} 不存在，跳过删除")
        result = {"acknowledged": True}
    return result


# ---------------------------------------------------------------------------
# 主流程：顺序执行所有操作
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    INDEX = "prerequisite_demo_chunks"

    print("=" * 60)
    print("Elasticsearch 前置知识 CRUD Demo")
    print("=" * 60)

    # 1. 连接
    print("\n--- 1. connect ---")
    es = connect()

    # 2. 创建索引
    print("\n--- 2. create_index ---")
    create_index(es, INDEX)

    # 3. 索引单条文档
    print("\n--- 3. index_document ---")
    index_document(es, INDEX)

    # 4. 批量索引
    print("\n--- 4. bulk_index ---")
    bulk_index(es, INDEX)

    # 5. 按 ID 获取文档
    print("\n--- 5. get_document ---")
    get_document(es, INDEX)

    # 6. match 全文搜索
    print("\n--- 6. search_match ---")
    search_match(es, INDEX)

    # 7. bool 组合查询
    print("\n--- 7. search_bool ---")
    search_bool(es, INDEX)

    # 8. 部分更新文档
    print("\n--- 8. update_document ---")
    update_document(es, INDEX)

    # 9. 删除单条文档
    print("\n--- 9. delete_document ---")
    delete_document(es, INDEX)

    # 10. 删除索引（清理 demo 数据）
    print("\n--- 10. delete_index ---")
    delete_index(es, INDEX)

    print("\n" + "=" * 60)
    print("Demo 全部执行完毕！")
    print("=" * 60)
