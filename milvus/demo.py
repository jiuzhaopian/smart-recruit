"""
Milvus 前置知识学习模块 - CRUD 完整演示

本脚本覆盖 Milvus 的 14 个核心操作，使用与 SmartRecruit 项目
(vector_store.py) 一致的 pymilvus 接口和字段设计。

运行前提：Milvus 服务已启动在 localhost:19530
依赖安装：pip install pymilvus numpy
"""

import time
import numpy as np
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
URI = "http://localhost:19530"
COLLECTION_NAME = "prerequisite_demo_collection"  # 不污染项目正式数据


# ==================== 1. 连接 ====================

def connect() -> MilvusClient:
    """创建 MilvusClient 并连接到 Milvus 服务。

    Args:
        uri: Milvus 服务地址，本项目使用 gRPC 协议 19530 端口。

    Returns:
        MilvusClient 实例，后续所有操作都通过它发起。
    """
    client = MilvusClient(uri="http://localhost:19530")
    print(f"[connect] 已连接到 {URI}")
    return client


# ==================== 2. 检查集合是否存在 ====================

def has_collection(client: MilvusClient) -> bool:
    """检查集合是否已存在。

    Args:
        client: MilvusClient 实例。

    Returns:
        True 表示集合已存在，False 表示不存在。
    """
    exists = client.has_collection("prerequisite_demo_collection")
    print(f"[has_collection] 集合 '{COLLECTION_NAME}' 存在: {exists}")
    return exists


# ==================== 3. 创建 Schema 和字段 ====================

def create_schema_and_fields():
    """创建集合的 Schema 并添加 5 种类型的字段。

    字段设计来源于 vector_store.py 的 _create_or_load_collection 方法
    （步骤 3.3 ~ 3.11），本项目实际有 8 个字段，这里精简为 5 个
    核心字段以突出类型差异。

    Returns:
        schema 对象，用于后续 create_collection。
    """
    # auto_id=False：主键由调用方生成（项目中用 chunk.metadata["id"]）
    # enable_dynamic_field=True：允许插入 schema 未定义的字段
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)


    # 字段 1：id — VARCHAR 主键
    # 项目中 max_length=100，这里用 64 简化
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)

    # 字段 2：dense_vector — 稠密向量（FLOAT_VECTOR）
    # 维度 1024 与 BGE-M3 的 dense 输出一致
    # 用于语义搜索，能捕获"意思相近但用词不同"的内容
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

    # 字段 3：sparse_vector — 稀疏向量（SPARSE_FLOAT_VECTOR）
    # 无需指定维度，长度由数据决定
    # 用于子词级别的精确匹配，与稠密向量互补
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 字段 4：text — 文本内容
    # 存储简历子块的原文，VARCHAR 最大 65535
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

    # 字段 5：gender — 标量字段，用于元数据过滤
    # vector_store.py 步骤 3.9，max_length=10
    schema.add_field(field_name="gender", datatype=DataType.VARCHAR, max_length=16)

    print("[create_schema_and_fields] Schema 创建完成，包含 5 个字段")
    return schema


# ==================== 4. 创建索引参数 ====================

def create_index_params(client: MilvusClient):
    """创建索引参数，为稠密向量和稀疏向量分别指定索引类型。

    与 vector_store.py 步骤 3.12 ~ 3.14 完全一致。

    Args:
        client: MilvusClient 实例。

    Returns:
        index_params 对象。
    """
    index_params = client.prepare_index_params()

    # 稠密向量索引：IVF_FLAT
    # - metric_type=IP：内积距离，向量归一化后等价于余弦相似度
    # - nlist=128：将向量空间分为 128 个 Voronoi 单元，搜索时只扫描
    #   nprobe 个最近单元，平衡精度与速度
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )

    # 稀疏向量索引：SPARSE_INVERTED_INDEX
    # - metric_type=IP：内积距离
    # - drop_ratio_build=0.2：建索引时丢弃最小的 20% 非零值，
    #   减少索引体积，对召回率影响极小
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.2},
    )

    print("[create_index_params] 索引参数创建完成（IVF_FLAT + SPARSE_INVERTED_INDEX）")
    return index_params


# ==================== 5. 创建集合 ====================

def create_collection(client: MilvusClient, schema, index_params):
    """使用 schema 和 index_params 创建集合。

    对应 vector_store.py 步骤 3.15。

    Args:
        client: MilvusClient 实例。
        schema: create_schema_and_fields() 返回的 schema。
        index_params: create_index_params() 返回的索引参数。
    """
    client.create_collection(
        collection_name="prerequisite_demo_collection",
        schema=schema,
        index_params=index_params,
    )
    print(f"[create_collection] 集合 '{COLLECTION_NAME}' 创建完成")


# ==================== 6. 加载集合到内存 ====================

def load_collection(client: MilvusClient):
    """将集合加载到内存，加载后才能执行搜索和查询。

    对应 vector_store.py 步骤 3.16。
    Milvus 2.x 的 MilvusClient.create_collection 内部会自动 load，
    但显式调用 load_collection 可确保集合已就绪。

    Args:
        client: MilvusClient 实例。
    """
    client.load_collection(COLLECTION_NAME)
    print(f"[load_collection] 集合 '{COLLECTION_NAME}' 已加载到内存")


# ==================== 7. 插入数据 ====================

def insert(client: MilvusClient):
    """插入 3 条简历子块示例数据。

    对应 vector_store.py 步骤 4.12 ~ 4.14。
    dense_vector 用 numpy 随机 1024 维模拟 BGE-M3 输出，
    sparse_vector 用 {index: value} 字典模拟稀疏编码。

    Args:
        client: MilvusClient 实例。

    Returns:
        插入结果（包含 insert_count 等）。
    """
    data = [
        {
            "id": "resume_001_chunk_001",
            "dense_vector": np.random.randn(1024).tolist(),
            "sparse_vector": {5001: 0.8, 20001: 0.6, 3001: 0.3},
            "text": "张三，男，北京大学计算机科学专业，5年后端开发经验，精通 Java、Python、MySQL。",
            "gender": "男",
        },
        {
            "id": "resume_001_chunk_002",
            "dense_vector": np.random.randn(1024).tolist(),
            "sparse_vector": {1200: 0.9, 8500: 0.5, 15000: 0.4},
            "text": "曾主导电商平台微服务架构改造，QPS 从 500 提升至 5000，获公司年度技术突破奖。",
            "gender": "男",
        },
        {
            "id": "resume_002_chunk_001",
            "dense_vector": np.random.randn(1024).tolist(),
            "sparse_vector": {3000: 0.7, 9000: 0.6, 22000: 0.5},
            "text": "李四，女，清华大学软件工程专业，3年数据分析经验，熟练使用 Spark、Flink。",
            "gender": "女",
        },
    ]

    result = client.insert("prerequisite_demo_collection", data)
    print(f"[insert] 插入 {len(data)} 条数据，结果: {result}")

    # 强制刷新，确保数据已写入存储并可被检索
    # Milvus 的写入是异步的，insert 返回后数据可能还未被索引
    client.flush(COLLECTION_NAME)
    time.sleep(1)  # 等待索引构建完成
    print(f"[insert] 数据已刷新并就绪")
    return result


# ==================== 8. 按 ID 获取数据 ====================

def get_by_id(client: MilvusClient, ids: list):
    """按主键 ID 获取完整文档。

    项目中使用场景：获取特定子块的原文和元数据。

    Args:
        client: MilvusClient 实例。
        ids: 要获取的 ID 列表。

    Returns:
        匹配的文档列表。
    """
    results = client.get(
        collection_name="prerequisite_demo_collection",
        ids=["resume_001_chunk_001"],
        output_fields=["id", "text", "gender"],
    )
    print(f"[get_by_id] 获取到 {len(results)} 条记录:")
    for r in results:
        print(f"  - id={r['id']}, gender={r['gender']}, text={r['text'][:50]}...")
    return results


# ==================== 9. 标量字段查询 ====================

def query(client: MilvusClient):
    """按标量字段条件查询（非向量搜索）。

    项目中用于元数据过滤：gender、age、work_experience 等字段。
    这就是"元数据过滤"——精确匹配，不涉及向量距离。

    Args:
        client: MilvusClient 实例。

    Returns:
        符合条件的文档列表。
    """
    results = client.query(
        collection_name="prerequisite_demo_collection",
        filter='gender == "男"',
        output_fields=["id", "text", "gender"],
    )
    print(f"[query] gender=='男' 的记录共 {len(results)} 条:")
    for r in results:
        print(f"  - id={r['id']}, text={r['text'][:50]}...")
    return results


# ==================== 10. 稠密向量单路搜索 ====================

def search_dense(client: MilvusClient):
    """仅使用稠密向量进行单路 ANN 搜索。

    这是 hybrid_search 的基础组件——理解 AnnSearchRequest 的参数
    是理解混合搜索的前提。

    Args:
        client: MilvusClient 实例。

    Returns:
        搜索结果列表。
    """
    query_vector = np.random.randn(1024).tolist()

    # AnnSearchRequest 参数说明：
    # - data: 查询向量列表（可以是批量），这里只有 1 条查询
    # - anns_field: 要搜索的向量字段名
    # - param: 搜索参数
    #   - metric_type: 距离度量，必须与建索引时一致（IP）
    #   - params.nprobe: 搜索时扫描的 Voronoi 单元数，
    #     nprobe 越大召回率越高但速度越慢
    # - limit: 返回 Top-K 条结果
    dense_req = AnnSearchRequest(
        data=[query_vector],  # 查询向量列表（可批量）
        anns_field="dense_vector",  # 搜索的向量字段
        param={  # 搜索参数
            "metric_type": "IP",  # 距离度量（必须与索引一致）
            "params": {"nprobe": 10},  # 搜索时扫描 10 个 Voronoi 单元
        },
        limit=3,  # 返回 Top-3
    )

    # 单路搜索用 hybrid_search 接口（只有 1 个 request）
    results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req],  # 只有 1 个 request
        ranker=WeightedRanker(1.0),  # 单路权重 1.0
        limit=3,
        output_fields=["id", "text", "gender"],  # 添加这行：指定返回的字段
    )

    print(f"[search_dense] 稠密向量搜索返回 {len(results[0])} 条结果:")
    for hit in results[0]:
        # 修复：hit 本身就是字典，可以直接访问字段
        print(f"  - id={hit['id']}, score={hit['distance']:.4f}, "
              f"text={hit['entity']['text'][:50]}...")
    return results


# ==================== 11. 稠密+稀疏混合搜索（核心） ====================

def hybrid_search(client: MilvusClient):
    """稠密向量 + 稀疏向量混合搜索。

    这是 SmartRecruit 的核心检索策略，对应 vector_store.py 步骤 4 中
    hybrid_search_with_rerank 方法的向量检索部分。

    原理：稠密向量捕获语义相似性（"后端开发"≈"服务端编程"），
    稀疏向量捕获词级精确匹配（"Java"必须命中）。两者加权融合，
    兼顾语义理解和精确召回。

    Args:
        client: MilvusClient 实例。

    Returns:
        混合搜索结果列表。
    """
    query_vector = np.random.randn(1024).tolist()
    # 稀疏查询向量：模拟 BGE-M3 sparse 编码的输出
    # key 是 token index，value 是权重
    sparse_query = {5001: 0.7, 1200: 0.9, 3000: 0.5}

    # --- 第一路：稠密向量搜索 ---
    dense_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="dense_vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=3,
    )

    # --- 第二路：稀疏向量搜索 ---
    # 注意：稀疏向量的 param 只需 metric_type，不需要 nprobe
    sparse_req = AnnSearchRequest(
        data=[sparse_query],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=3,
    )

    # --- 加权融合 ---
    # WeightedRanker(0.7, 0.3) 含义：
    #   - 第 1 路（稠密）权重 0.7：语义匹配占主导
    #   - 第 2 路（稀疏）权重 0.3：词级匹配作为补充
    # 融合公式：final_score = 0.7 * dense_score + 0.3 * sparse_score
    # 项目中也是这个比例（vector_store.py 第 264 行）
    # 加权融合：稠密 0.7 + 稀疏 0.3
    results = client.hybrid_search(
        collection_name="prerequisite_demo_collection",
        reqs=[dense_req, sparse_req],
        ranker=WeightedRanker(0.7, 0.3),
        limit=3,
        output_fields=["id", "text", "gender"],  # 添加这行：指定返回的字段
    )

    print(f"[hybrid_search] 混合搜索返回 {len(results[0])} 条结果:")
    for hit in results[0]:
        print(f"  - id={hit['id']}, score={hit['distance']:.4f}, "
              f"text={hit['entity']['text'][:50]}...")
    return results


# ==================== 12. 更新或插入（upsert） ====================

def upsert(client: MilvusClient):
    """按 ID 更新已有记录，如果 ID 不存在则插入新记录。

    项目中用于更新简历子块内容（重新解析后覆盖旧数据）。

    Args:
        client: MilvusClient 实例。

    Returns:
        upsert 结果。
    """
    new_vector = np.random.randn(1024).tolist()
    result = client.upsert(COLLECTION_NAME, [
        {
            "id": "resume_001_chunk_001",  # 已存在则更新，不存在则插入
            "dense_vector": new_vector,
            "sparse_vector": {5001: 0.9, 20001: 0.7, 3001: 0.4},
            "text": "张三，男，北京大学计算机科学专业，6年后端开发经验（更新），精通 Java、Python、Go、MySQL。",
            "gender": "男",
        },
    ])
    print(f"[upsert] 更新结果: {result}")
    return result


# ==================== 13. 按条件删除 ====================

def delete(client: MilvusClient):
    """按过滤条件删除记录。

    对应 vector_store.py 中按 doc_hash 删除简历所有子块的场景
    （delete by filter: doc_hash == 'xxx'）。

    Args:
        client: MilvusClient 实例。

    Returns:
        删除结果。
    """
    result = client.delete(
        collection_name="prerequisite_demo_collection",
        filter='id == "resume_002_chunk_001"',
    )
    print(f"[delete] 删除结果: {result}")
    return result


# ==================== 14. 释放集合 ====================

def release_collection(client: MilvusClient):
    """将集合从内存释放，释放后无法搜索，但数据仍在磁盘上。

    与 load_collection 互为逆操作。项目中不会主动 release，
    但在 demo 中演示以说明内存管理机制。

    Args:
        client: MilvusClient 实例。
    """
    client.release_collection(COLLECTION_NAME)
    print(f"[release_collection] 集合 '{COLLECTION_NAME}' 已从内存释放")


# ==================== 主流程 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Milvus CRUD 完整演示 - SmartRecruit 前置知识")
    print("=" * 60)

    # 1. 连接
    client = connect()

    # 2. 清理旧数据（如果集合已存在，先删除以保证 demo 可重复运行）
    if has_collection(client):
        client.drop_collection(COLLECTION_NAME)
        print(f"[cleanup] 已删除旧集合 '{COLLECTION_NAME}'")

    # 3. 创建 Schema
    schema = create_schema_and_fields()

    # 4. 创建索引参数
    index_params = create_index_params(client)

    # 5. 创建集合
    create_collection(client, schema, index_params)

    # 6. 加载到内存
    load_collection(client)

    # 7. 插入数据
    insert(client)

    # 8. 按 ID 获取
    get_by_id(client, ["resume_001_chunk_001", "resume_002_chunk_001"])

    # 9. 标量查询
    query(client)

    # 10. 稠密向量搜索
    search_dense(client)

    # 11. 混合搜索（核心）
    hybrid_search(client)

    # 12. 更新
    upsert(client)

    # 13. 删除
    delete(client)

    # 14. 释放
    release_collection(client)

    # 清理：删除 demo 集合
    client.drop_collection(COLLECTION_NAME)
    print(f"\n[cleanup] 已清理 demo 集合 '{COLLECTION_NAME}'")

    print("=" * 60)
    print("全部 14 个操作演示完成！")
    print("=" * 60)
