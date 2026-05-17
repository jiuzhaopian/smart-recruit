# rag/chain.py
import asyncio
import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from config import config
from utils.vector_store import VectorStore

# 初始化LLM和检索器
llm = ChatOpenAI(
    model_name="qwen3.6-plus",
    openai_api_key=config.DASHSCOPE_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1
)
retriever = VectorStore()

#查询改写的提示词
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    你是一位专业的招聘助理。请根据对话历史和用户的最新输入，
    将用户的最新需求改写为一个清晰、独立、适合检索的查询语句。
    要求：
    1. 结合对话历史补充上下文信息（如之前提到的筛选条件）
    2. 消除代词指代（"他""那个人"→具体的人名或条件）
    3. 输出只有改写后的查询，不要解释
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])


#获得简历后组织语言回答用户的提示词
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
作为资深技术招聘官和最终审核人，根据【用人需求】和【简历上下文】推荐最匹配的候选人。

【简历上下文】:
---
{context}
---

输出要求：
1. **最终审核**：作为最后一道关卡，确保简历与【用人需求】高度相关。
2. **精准推荐**：仅当简历高度匹配时，生成推荐理由；忽略勉强相关或无关的简历。
3. **JSON 格式**：严格按以下格式输出可能为空的列表，不添加额外解释：
    [
        {{
            "candidate_id": 1,
            "reason": "推荐理由...",
            "file_path": "简历文件名.pdf",
            "doc_hash": "简历哈希值"
        }},
        ...
    ]
    """),
    ("user", "【用人需求】: {input}"),
])


#对返回简历的格式化
def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "未在简历库中找到相关信息。"
    formatted_docs = []
    for doc in docs:
        doc_hash = doc.metadata.get("doc_hash", doc.metadata.get("hash", "N/A"))
        doc_str = (
            f"简历来源文件: {os.path.basename(doc.metadata.get('file_path', 'N/A'))}\n"
            f"简历哈希值: {doc_hash}\n"
            f"内容: {doc.page_content}"
        )
        formatted_docs.append(doc_str)
    return "\n\n---\n\n".join(formatted_docs)


async def get_rag_chain():
    """构建并返回支持历史记录和动态参数的异步 RAG 链。"""
    #异步调用
    async def retrieve_and_format_context(input_dict: dict, config: RunnableConfig) -> str:
        rewritten_question = await(REWRITE_PROMPT | llm | StrOutputParser()).ainvoke({
            "input":input_dict["input"],
            "chat_history":input_dict["chat_history"],
        },config=config)
        logger.info(f"原始查询: '{input_dict['input']}' -> 改写后查询: '{rewritten_question}'")


        params = input_dict.get("params",{})
        #异步执行高级混合检索
        try:
            retrieved_docs = await retriever.aget_relevant_documents(
                query=rewritten_question,
                params=params
            )
        except Exception as e:
            logger.error(f"检索器执行失败: {e}", exc_info=True)
            return "检索简历时发生内部错误，请稍后再试。"

        #格式化
        context = _format_docs(retrieved_docs)
        logger.debug(f"为LLM准备的上下文: \n{context}")
        return context

    #简化的短chain
    rag_chain = (
        RunnablePassthrough.assign(context=retrieve_and_format_context)
                ) | ANSWER_PROMPT | llm | StrOutputParser()
    return rag_chain


# --- [新增] 验证代码 ---
if __name__ == '__main__':
    async def main():
        """独立验证RAG Chain的核心功能"""
        logger.info("=" * 50)
        logger.info("开始独立验证 chain.py 模块...")

        rag_chain = await get_rag_chain()

        # --- 测试用例 1: 简单的招聘需求 ---
        print("\n--- 测试用例 1: 简单招聘需求 ---")
        query1 = "我需要一个懂AI算法的工程师"
        params1 = {"count": 2}
        print(f"用户: {query1}, 参数: {params1}")

        response1 = await rag_chain.ainvoke({
            "input": query1,
            "chat_history": [],
            "params": params1
        })
        print(f"AI响应 (部分): {response1[:300]}...")
        assert response1 and isinstance(response1, str)
        print("【测试用例 1 通过】")

        # --- 测试用例 2: 带筛选条件的招聘需求 ---
        print("\n--- 测试用例 2: 带筛选条件的招聘需求 ---")
        query2 = "帮我找一个有5年以上经验的算法工程师"
        params2 = {"count": 1, "experience_min": 5}
        print(f"用户: {query2}, 参数: {params2}")

        response2 = await rag_chain.ainvoke({
            "input": query2,
            "chat_history": [],
            "params": params2
        })
        print(f"AI响应 (部分): {response2[:300]}...")
        assert response2 and isinstance(response2, str)
        print("【测试用例 2 通过】")

        logger.info("=" * 50)
        logger.success("chain.py 模块所有功能验证通过！")


    asyncio.run(main())


