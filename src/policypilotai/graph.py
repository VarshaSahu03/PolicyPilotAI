
import os

from dotenv import load_dotenv
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from policypilotai.guardrails import (
    check_input_guardrail,
    check_output_guardrail
)

from policypilotai.llm_gateway import stream_with_fallback

from policypilotai.hybrid_rag.retriever import hybrid_retrieve


# ==========================================
# ENVIRONMENT
# ==========================================

load_dotenv()


# ==========================================
# GRAPH STATE
# ==========================================

class State(TypedDict, total=False):

    question: str
    context: str
    answer: str


# ==========================================
# 1. RETRIEVE NODE
# ==========================================

def retrieve(state: State):

    question = state["question"]

    chunks = hybrid_retrieve(
        question,
        top_k=5,
        alpha=0.5
    )

    if not chunks:

        return {
            "context": "No relevant policy found."
        }

    context = "\n\n".join(
        chunk if isinstance(chunk, str)
        else chunk.page_content
        for chunk in chunks
    )

    return {
        "context": context
    }


# ==========================================
# 2. EXTRACT TEXT FROM LLM CHUNKS
# ==========================================

def extract_chunk_text(chunk):

    # Plain string
    if isinstance(chunk, str):
        return chunk

    # LangChain AIMessageChunk
    content = getattr(chunk, "content", None)

    if isinstance(content, str):
        return content

    # Multimodal/list content
    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":
                    text_parts.append(
                        block.get("text", "")
                    )

            elif isinstance(block, str):
                text_parts.append(block)

        return "".join(text_parts)

    return ""


# ==========================================
# 3. GENERATE NODE
# ==========================================

def generate(
    state: State,
    config: RunnableConfig
):

    question = state["question"]
    context = state["context"]

    task_type = (
        "complex"
        if len(question.split()) > 15
        else "simple"
    )

    prompt = ChatPromptTemplate.from_template("""
    You are PolicyPilotAI, an internal HR policy assistant.
    Answer questions strictly using the retrieved company policy documents.
    If the answer is not available in the provided context,
    clearly state that the policy documents do not specify it.
    Keep answers professional, concise and easy to read.

    You are a professional AI assistant. Your task is to provide
    clear, structured, natural, and visually readable responses.

    STRICT FORMATTING RULES:

    1. INTRODUCTION
    - Start with a natural introductory paragraph.
    - Directly answer the user's question.
    - Keep the introduction short and meaningful.
    - Do not start with unnecessary greetings.

    2. MAIN HEADINGS
    - Use **bold text** for main section headings.
    - Keep headings short and meaningful.
    - Leave one blank line before and after each heading.
    - Do not use headings for every small explanation.

    3. BULLET POINTS
    - Use hyphen (-) bullet points for explanations,
    features, benefits, and important information.
    - Start each bullet with a short, meaningful phrase in bold.
    - Follow the bold phrase with a clear explanation.
    - Keep each bullet focused on one idea.
    - Maintain consistent indentation.

    4. NESTED BULLET POINTS
    - Use nested bullets only when additional details
    belong to a parent bullet.
    - Keep nested explanations properly indented.
    - Avoid unnecessary nesting.

    5. NUMBERED LISTS
    - Use numbered lists only for sequential steps,
    procedures, instructions, or ordered processes.
    - Never number ordinary explanations or features.

    6. PARAGRAPHS
    - Keep paragraphs short and easy to understand.
    - Separate paragraphs with one blank line.
    - Avoid large blocks of continuous text.

    7. BOLD FORMATTING
    - Use bold text for important terms, key phrases,
    section headings, and the beginning of bullet points.
    - Do not bold entire paragraphs.
    - Avoid excessive bold formatting.

    8. TABLES
    - Use Markdown tables only when comparing multiple items
    or presenting structured information.
    - Keep tables simple and readable.

    9. RESPONSE STRUCTURE
    - Use only the sections necessary for the question.
    - Do not force headings into short answers.
    - Avoid unnecessary numbering.
    - Do not repeat information.
    - Do not add decorative symbols or excessive emojis.
    - End naturally without repeating the entire answer.

    10. MARKDOWN RULES
    - Return valid Markdown only.
    - Use **text** for bold formatting.
    - Use - for unordered bullet points.
    - Use 1. for numbered lists.
    - Use proper Markdown indentation for nested lists.
    - Never use HTML tags.
    - Never use manual spaces to align text.
    - Always put a blank line between headings,
    paragraphs, and lists.

    11. CONTENT QUALITY
    - Answer the actual question directly.
    - Explain technical concepts in simple language.
    - Be concise but sufficiently detailed.
    - Do not invent facts.
    - Preserve the meaning of source documents.

    IMPORTANT:
    The response should resemble a professionally written
    AI assistant answer with clean paragraphs, bold headings,
    and properly aligned bullet points.

    Do not force every response into the same template.
    Choose the structure according to the user's question.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
        )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    # --------------------------------------
    # COLLECT LLM STREAM
    # --------------------------------------

    answer_chunks = []
    full_answer = ""

    for chunk in stream_with_fallback(
        messages,
        task_type=task_type,
        config=config
    ):

        text = extract_chunk_text(chunk)

        if text:
            answer_chunks.append(text)
            full_answer += text

    full_answer = full_answer.strip()

    if not full_answer:

        return {
            "answer": "I could not generate an answer. Please try again."
        }

    # --------------------------------------
    # OUTPUT GUARDRAIL
    # --------------------------------------

    guardrail_result = check_output_guardrail(full_answer)

    if isinstance(guardrail_result, tuple):

        is_safe = guardrail_result[0]
        rejection_message = (
            guardrail_result[1]
            if len(guardrail_result) > 1
            else "The generated response could not be shared."
        )

    else:

        is_safe = bool(guardrail_result)
        rejection_message = (
            "The generated response could not be shared."
        )

    if not is_safe:

        full_answer = str(rejection_message)
        answer_chunks = [full_answer]

    # --------------------------------------
    # STREAM VALIDATED ANSWER
    # --------------------------------------

    writer = get_stream_writer()

    for text in answer_chunks:

        if text:
            writer(text)

    return {
        "answer": full_answer
    }


# ==========================================
# 4. BUILD LANGGRAPH
# ==========================================

builder = StateGraph(State)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.set_entry_point("retrieve")

builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


# ==========================================
# 5. STREAM USER QUESTION
# ==========================================

async def stream_user_question(question: str):

    # --------------------------------------
    # INPUT GUARDRAIL
    # --------------------------------------

    is_safe, message = check_input_guardrail(question)

    if not is_safe:

        yield str(message)
        return

    # --------------------------------------
    # GRAPH STREAMING
    # --------------------------------------

    async for chunk in graph.astream(
        {"question": question},
        stream_mode="custom"
    ):

        if isinstance(chunk, str) and chunk:
            yield chunk
