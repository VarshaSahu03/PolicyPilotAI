
import os
from langchain_groq import ChatGroq


def get_llm(task_type="simple", streaming=False):

    if task_type == "simple":
        model = "qwen/qwen3.8-27b"

    elif task_type == "complex":
        model = "qwen/qwen3.8-27b"

    elif task_type == "evaluation":
        model = "llama-3.1-8b-instant"

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return ChatGroq(
        model=model,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        streaming=streaming
    )


def call_with_fallback(
    prompt_value,
    task_type="simple",
    config=None
):
    llm = get_llm(task_type, streaming=False)

    try:
        return llm.invoke(prompt_value, config=config)

    except Exception as e:
        print(f"Primary LLM failed: {e}. Switching to backup.")

        backup_llm = get_llm("simple", streaming=False)

        return backup_llm.invoke(
            prompt_value,
            config=config
        )


def stream_with_fallback(
    prompt_value,
    task_type="simple",
    config=None
):
    llm = get_llm(task_type, streaming=True)

    try:
        yield from llm.stream(prompt_value, config=config)

    except Exception as e:
        print(f"Primary LLM failed: {e}. Switching to backup.")

        backup_llm = get_llm("simple", streaming=True)

        yield from backup_llm.stream(
            prompt_value,
            config=config
        )
