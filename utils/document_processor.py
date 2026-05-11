# document_processor.py
import os
import re
import hashlib
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader, Docx2txtLoader, \
    UnstructuredPowerPointLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from datetime import datetime
from openai import OpenAI
import base64
from loguru import logger
from config import config
import json

# --- 日志配置 ---
logger.add(os.path.join(config.LOG_DIR, "document_processor.log"), rotation="10 MB", encoding="utf-8")

# --- LLM 客户端初始化 ---
# 为结构化数据解析创建一个独立的客户端
parser_client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def compute_file_hash(file_path: str) -> str:
    """
    计算文件的 MD5 哈希值，用于简历去重。

    原理：同一份文件（内容完全相同）的 MD5 值一定相同，
    不同文件的 MD5 值几乎一定不同（碰撞概率极低）。
    因此可以用 MD5 来判断"这份简历我们是不是已经处理过了"。

    Args:
        file_path: 文件路径

    Returns:
        32位十六进制 MD5 字符串，例如 "d41d8cd98f00b204e9800998ecf8427e"

    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    hasher = hashlib.md5()
    try:
        # 以二进制模式读取，避免编码问题
        with open(file_path, "rb") as f:
            # 每次读 4KB，避免大文件一次性占满内存
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"计算文件hash失败: {file_path}, 错误: {str(e)}")
        raise


def extract_text_from_image(image_path: str, client: OpenAI) -> str:
    """
    使用多模态大模型从图片中提取简历文本。

    本项目使用阿里云 qwen-omni-turbo 模型，它是一个视觉语言模型（VLM），
    能够理解图片中的文字和布局。

    与传统 OCR（如 Tesseract）的区别：
    - OCR 只能识别文字，不理解语义
    - VLM 不仅能识别文字，还能理解布局（如"这段是工作经历"）

    Args:
        image_path: 图片文件路径（.jpg / .png）
        client: OpenAI 兼容客户端（已配置 DashScope base_url）

    Returns:
        提取的纯文本字符串

    Raises:
        FileNotFoundError: 图片文件不存在
        Exception: API 调用失败
    """
    # 1. 将图片转为 base64 编码
    #    OpenAI Vision API 要求图片以 base64 格式嵌入请求
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # 2. 构造多模态请求
    #    消息体是一个列表，包含文本和图片两种类型的 content
    response = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "提取图片中的简历文本信息，包括个人信息、教育背景、工作经历等。输出纯文本。",
                    },
                ],
            }
        ],
        stream=False,  # 不使用流式输出，直接返回完整结果
    )

    content = response.choices[0].message.content
    logger.info(f"图片提取文本成功: {image_path}, 内容长度: {len(content)}")
    return content


# --- 文件格式 → 加载器映射表 ---
# None 表示不使用 LangChain Loader，走特殊处理（图片用 VLM）
document_loaders = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".jpg": None,   # 图片：使用 extract_text_from_image()
    ".png": None,
    ".md": UnstructuredMarkdownLoader,
}

def load_and_hash_document(file_path: str, client: OpenAI) -> tuple[str, str]:
    """
    加载文件内容并计算 MD5 哈希。

    根据文件扩展名自动选择对应的加载器：
    - 图片（.jpg/.png）：调用 qwen-omni-turbo 多模态提取
    - .txt：尝试 utf-8 → gbk → latin1 三种编码（兼容 Windows GBK 文件）
    - 其他格式：使用对应的 LangChain Loader

    Args:
        file_path: 文件路径
        client: OpenAI 兼容客户端（图片提取时需要）

    Returns:
        (content, doc_hash) 元组
        - content: 提取的纯文本
        - doc_hash: 文件 MD5 哈希（32位十六进制）

    Raises:
        ValueError: 不支持的文件格式
        UnicodeDecodeError: txt 文件所有编码都失败
        Exception: 加载过程中的其他错误
    """
    logger.info(f"开始加载并哈希文件: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()

    # 1. 检查格式是否支持
    if file_extension not in document_loaders:
        raise ValueError(f"不支持的文件类型: {file_extension}")

    content = ""

    try:
        # 2. 图片走多模态提取
        if file_extension in [".jpg", ".png"]:
            content = extract_text_from_image(file_path, client)
        else:
            loader_class = document_loaders[file_extension]

            # 3. TXT 文件需要尝试多种编码（兼容中文环境）
            if file_extension == ".txt":
                encodings = ["utf-8", "gbk", "latin1"]
                for enc in encodings:
                    try:
                        loader = loader_class(file_path, encoding=enc)
                        content = loader.load()[0].page_content
                        break  # 成功就用这个编码
                    except UnicodeDecodeError:
                        continue  # 当前编码失败，试下一个
                else:
                    # 所有编码都失败
                    raise UnicodeDecodeError(
                        f"无法以支持的编码加载文件: {file_path}",
                        b"", 0, 0, "尝试所有编码失败"
                    )
            else:
                # 4. 其他格式直接加载
                loader = loader_class(file_path)
                content = loader.load()[0].page_content

        # 5. 计算文件 MD5
        doc_hash = compute_file_hash(file_path)
        logger.info(f"文件加载并哈希成功: {file_path}, hash: {doc_hash}")
        return content, doc_hash

    except Exception as e:
        logger.error(f"加载或哈希文件失败: {file_path}, 错误: {str(e)}")
        raise


# --- 简历结构化信息提取的 Prompt 模板 ---
RESUME_PARSER_PROMPT = """
你是一个顶级的HR简历分析专家。请从以下简历文本中，提取出关键的结构化信息。

**严格遵守以下规则:**
1.  **提取字段**: 只提取以下字段：`name` (姓名), `gender` (性别), `age` (年龄), `work_experience` (工作年限)。
2.  **JSON格式**: 必须严格按照JSON格式输出，不要有任何额外的解释或Markdown标记。
3.  **逻辑推断**:
    - **姓名 (name)**: 通常是文本开头最明显的人名。
    - **性别 (gender)**: 从文本中明确的"男"或"女"字样判断。如果未提及，则为 "未提供"。
    - **年龄 (age)**: 根据出生年份、或直接描述的年龄计算。例如"1990年出生"在2024年应计算为34岁。如果无法推断，则为 -1。
    - **工作年限 (work_experience)**: 根据工作经历的总时长计算。例如"2020年7月至2023年7月"是3年。如果无法推断，则为 -1。
4.  **数值类型**: `age` 和 `work_experience` 必须是整数。

**简历文本:**
---
{resume_text}
---

**输出JSON:**
"""


def parse_resume_structure(resume_text: str, client: OpenAI) -> Dict[str, Any]:
    """
    使用 LLM 从简历纯文本中提取结构化信息。

    为什么不用正则/规则解析？
    - 简历格式千变万化："男，30岁" vs "Gender: Male, 1994" vs 只写毕业年份不写年龄
    - 规则永远覆盖不全，LLM 能"理解"语义并灵活推断

    提取的字段会作为元数据存入向量数据库（Milvus），
    用于检索时的精确过滤（如"只要男性""年龄≤35""5年以上经验"）。

    Args:
        resume_text: 简历纯文本
        client: OpenAI 兼容客户端

    Returns:
        dict，包含 name/gender/age/work_experience 四个字段
        解析失败时返回默认值 {"name": "未知", "gender": "未提供", "age": -1, "work_experience": -1}
    """
    logger.info("开始使用LLM解析简历结构化信息...")
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个顶级的HR简历分析专家。"},
                {"role": "user", "content": RESUME_PARSER_PROMPT.format(resume_text=resume_text)},
            ],
            temperature=0.0,  # 温度设为 0，保证输出稳定可复现
        )
        content = response.choices[0].message.content
        logger.debug(f"LLM原始解析结果: {content}")

        # 清理 LLM 输出：去掉可能的 ```json ``` 包裹
        json_str = content.strip().removeprefix("```json").removesuffix("```").strip()
        structured_data = json.loads(json_str)

        # 数据清洗：确保类型正确
        structured_data['age'] = int(structured_data.get('age', -1))
        structured_data['work_experience'] = int(structured_data.get('work_experience', -1))
        structured_data['gender'] = structured_data.get('gender', '未提供')

        logger.info(f"简历结构化信息解析成功: {structured_data}")
        return structured_data

    except Exception as e:
        logger.error(f"解析简历结构化信息失败: {e}", exc_info=True)
        # 优雅降级：返回默认值，不中断整个处理流程
        return {"name": "未知", "gender": "未提供", "age": -1, "work_experience": -1}


def process_document(doc: Document) -> List[Document]:
    """
    对单个文档进行父子块切分。

    切分策略：
    1. 先用父切分器切成大块（1000 字符）
    2. 再对每个父块用子切分器切成小块（400 字符）
    3. 每个子块携带父块的 ID 和完整内容（用于检索后返回上下文）

    切分器选择：
    - .md 文件使用 MarkdownTextSplitter（按标题切分，保持语义完整）
    - 其他格式使用 RecursiveCharacterTextSplitter（按字符逐级切分）

    Args:
        doc: LangChain Document 对象，必须包含 metadata['hash'] 和 metadata['file_path']

    Returns:
        List[Document]：子块列表，每个子块的 metadata 包含：
        - chunk_id: 子块唯一标识（格式：doc_{hash}_parent_{j}_child_{k}）
        - parent_id: 父块 ID
        - parent_content: 父块完整文本（检索命中后直接返回给 LLM，无需二次查询）
        - hash: 文件 MD5
        - file_path: 文件路径
        以及其他从原始 doc 继承的 metadata
    """
    logger.info(f"开始处理单个文档: {doc.metadata.get('file_path', 'N/A')}")

    # --- 初始化切分器 ---
    # 从 config 中读取参数，方便统一调整
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,  # 1000
        chunk_overlap=config.CHUNK_OVERLAP,   # 100
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,   # 400
        chunk_overlap=config.CHUNK_OVERLAP,   # 100
    )
    markdown_parent_splitter = MarkdownTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    markdown_child_splitter = MarkdownTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # --- 根据文件类型选择切分器 ---
    file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
    is_markdown = file_extension == ".md"
    parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
    child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
    logger.info(
        f"使用切分器: {'Markdown' if is_markdown else 'RecursiveCharacter'}"
    )

    # --- 第一步：切成父块 ---
    parent_docs = parent_splitter_to_use.split_documents([doc])
    logger.debug(f"切分为 {len(parent_docs)} 个父块")

    # --- 第二步：对每个父块切成子块，并注入元数据 ---
    child_chunks = []
    for j, parent_doc in enumerate(parent_docs):
        # 生成父块 ID
        parent_id = f"doc_{doc.metadata['hash']}_parent_{j}"

        # 给父块也加上标识（父块本身不入库）;这里也可不加
        #parent_doc.metadata["chunk_id"] = parent_id
        #parent_doc.metadata["page_content"] = parent_doc.page_content
        #parent_doc.metadata.update(doc.metadata) # 继承原始文档的 metadata

        # 将父块切成子块
        sub_chunks = child_splitter_to_use.split_documents([parent_doc])
        for k, sub_chunk in enumerate(sub_chunks):
            # 生成子块 ID
            chunk_id = f"{parent_id}_child_{k}"

            # 注入元数据
            sub_chunk.metadata["parent_id"] = parent_id
            sub_chunk.metadata["parent_content"] = parent_doc.page_content
            sub_chunk.metadata["chunk_id"] = chunk_id
            sub_chunk.metadata["id"] = chunk_id  # 兼容字段
            sub_chunk.metadata.update(doc.metadata)  # 继承原始文档的 metadata
            child_chunks.append(sub_chunk)
            logger.debug(
                f"生成子块: {chunk_id}, 父块: {parent_id}, "
                f"内容长度: {len(sub_chunk.page_content)}"
            )

    logger.info(f"文档 {doc.metadata['file_path']} 共生成 {len(child_chunks)} 个子块")
    return child_chunks


if __name__ == "__main__":
    """验证文档加载、解析和切分功能"""
    logger.info("=" * 50)
    logger.info("开始独立验证 document_processor.py 模块...")

    # 选择一个测试文件
    test_dir = config.LOCAL_RESUME_DIR
    test_file_name = "李明AI大模型产品经理简历.pdf"  # 可换成任意存在的文件名
    test_file_path = os.path.join(test_dir, test_file_name)

    if not os.path.exists(test_file_path):
        logger.error(f"测试文件不存在，请确保 '{test_file_path}' 存在后再运行验证。")
    else:
        try:
            # 1. 验证加载和哈希
            logger.info(f"--- 1. 测试加载与哈希 ---")
            content, doc_hash = load_and_hash_document(test_file_path, parser_client)
            assert content and doc_hash
            logger.info(f"加载成功: hash={doc_hash}, 内容长度={len(content)}")

            # 2. 验证结构化解析
            logger.info(f"--- 2. 测试结构化信息解析 ---")
            structured_data = parse_resume_structure(content, parser_client)
            assert isinstance(structured_data, dict) and "name" in structured_data
            logger.info(f"解析成功: {structured_data}")

            # 3. 验证切块
            logger.info(f"--- 3. 测试文档切块 ---")
            doc = Document(page_content=content, metadata={"file_path": test_file_path, "hash": doc_hash})
            chunks = process_document(doc)
            assert chunks and isinstance(chunks, list)
            logger.info(f"切块成功: 共生成 {len(chunks)} 个子块。")
            logger.info(f"第一个子块内容: {chunks[0].page_content}")
            logger.info(f"第一个子块元数据: {chunks[0].metadata}")

            logger.success("document_processor.py 模块所有功能验证通过！")
            print("\n[SUCCESS] document_processor.py module validation passed!")

        except Exception as e:
            logger.critical(f"document_processor.py 模块验证失败: {e}", exc_info=True)
            print(f"\n[FAILURE] document_processor.py module validation failed.")