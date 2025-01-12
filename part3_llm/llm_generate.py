# -*- coding: utf-8 -*-
"""
行为预测模型生成模块
该模块基于用户的手语动作和面部表情信息，生成行为预测结果
使用 LangChain 框架实现 LLM 推理
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import re
import datetime
import random
import json

from sentence_transformers import SentenceTransformer, util
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.output_parsers import (
    StructuredOutputParser, 
    ResponseSchema
)
from langchain_core.messages import AIMessage
from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Qdrant
from document import Document


class GenerationRequest(BaseModel):
    """请求参数模型类"""
    topic: str
    role: Optional[str] = None  # 角色设定，可选
    title: Optional[str] = None  # 标题，可选
    question: str              # 问题内容
    dialog: Optional[str] = None  # 对话内容，可选
    mode: Optional[str] = None    # 模式设置，可选


def initialize_embeddings() -> HuggingFaceEmbeddings:
    """
    初始化并加载embedding模型
    
    Returns:
        HuggingFaceEmbeddings: 加载好的embedding模型实例
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="model/embedding/m3e-base",
        model_kwargs={"device": "cpu"}
    )
    print("Embedding model loaded:", embeddings.model_name)
    return embeddings


def initialize_vector_store(embeddings: HuggingFaceEmbeddings) -> Qdrant:
    """
    初始化向量存储
    
    Args:
        embeddings: 已加载的embedding模型
        
    Returns:
        Qdrant: 初始化好的向量存储实例
    """
    docs_preload = [
        Document(
            page_content=str("hello"),
            metadata={"label": "role_comments"},
        )
    ]
    
    vector_store = Qdrant.from_documents(
        documents=docs_preload,
        embedding=embeddings,
        location=":memory:",
        collection_name="preload_docs",
    )
    return vector_store


def create_chat_prompt() -> ChatPromptTemplate:
    """
    创建对话提示模板
    
    Returns:
        ChatPromptTemplate: 配置好的提示模板
    """
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
            你现在是一个专业的行为预测分析专家，
            我将提供有关用户的两部分描述信息：
            1. 通过 Mediapipe 的手部关键点检测，提供用户的手语动作信息
            2. 通过 DeepFace 的面部表情识别，提供用户的面部表情信息
            {context}
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])


def model_generate(hand_info: str, emo_info: str) -> Dict[str, str]:
    """
    根据手势和表情信息生成行为预测结果
    
    Args:
        hand_info: 手语动作信息
        emo_info: 面部表情信息
        
    Returns:
        Dict[str, str]: 包含预测结果的字典
    """
    # 初始化模型和组件
    chat_model = ChatOllama(model="qwen2.5", device="cuda")
    embeddings = initialize_embeddings()
    vector_store = initialize_vector_store(embeddings)
    
    # 构建检索链
    vector_retriever = vector_store.as_retriever()
    history_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", """需求的描述是{input}"""),
        ("user", "Given the introduction in docs, generate answer in corresponding view"),
    ])
    
    # 创建检索链和文档链
    history_chain = create_history_aware_retriever(
        llm=chat_model,
        prompt=history_prompt,
        retriever=vector_retriever,
    )
    
    doc_prompt = create_chat_prompt()
    documents_chain = create_stuff_documents_chain(chat_model, doc_prompt)
    retriever_chain = create_retrieval_chain(history_chain, documents_chain)
    
    # 配置输出解析器
    response_schema = [
        ResponseSchema(
            name="answer",
            description="用尽量简短的信息描述用户下一步的行为预测结果",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    
    # 构建输入提示
    model_prompt = f"""
    请根据以下信息，生成对应的行为预测结果：
    用户的手语动作信息：{hand_info}，
    用户的面部表情信息：{emo_info}。
    """
    
    format_instruction = output_parser.get_format_instructions()
    prompt_template = PromptTemplate.from_template(
        template="""
        {format_prompt}
        please strictly answer in format: 
        {format_instruction}
        """,
        partial_variables={"format_instruction": format_instruction},
    )
    
    # 生成结果
    prompt_str_input = prompt_template.format(format_prompt=model_prompt)
    output_completion = retriever_chain.invoke({
        "input": prompt_str_input, 
        "chat_history": []
    })
    
    # 处理输出结果
    content = output_completion["answer"]
    json_str = content.strip("```json\n").strip("```")
    content_processed = re.sub(r'[{}""" ]+', "", json_str)
    
    start_index = content_processed.find("answer:") + len("answer:")
    answer_part = content_processed[start_index:].strip()
    
    return {"answer_part": answer_part}


if __name__ == "__main__":
    # 测试代码
    result = model_generate("ok的手势", "开心")
    print("预测结果:", result)