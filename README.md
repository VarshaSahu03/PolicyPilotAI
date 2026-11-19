# PolicyPilotAI

### AI-Powered HR Policy Assistant Using Hybrid RAG and LangGraph

PolicyPilotAI is an AI-powered application designed to answer employee queries based on HR policy documents. It uses Hybrid Retrieval-Augmented Generation (RAG) to retrieve relevant information from policy documents and generate context-aware answers using Large Language Models.

The application combines semantic search, keyword-based retrieval, LLM-powered response generation, and guardrails to improve the relevance and reliability of answers.

## Key Features

- **Hybrid Retrieval:** Combines vector similarity search and BM25 keyword search to retrieve relevant policy information.
- **LangGraph Workflow:** Organizes document retrieval and answer generation into a structured workflow.
- **LLM Gateway:** Supports task-based model selection and fallback handling.
- **Input and Output Guardrails:** Validates user queries and generated responses.
- **Streaming API:** Supports streaming responses from the FastAPI backend.
- **React Chat Interface:** Provides an interactive interface with formatted Markdown responses.
- **Response Evaluation:** Includes an evaluation module to assess generated answer quality.

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React.js, Vite, JavaScript |
| Backend | Python, FastAPI |
| AI Orchestration | LangChain, LangGraph |
| Retrieval | Hybrid RAG, BM25 |
| Vector Database | ChromaDB |
| Language Models | Groq |
| Package Management | uv, npm |

## Project Architecture

```text
PolicyPilotAI/
│
├── src/
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.jsx
│   │   │   ├── App.css
│   │   │   └── MarkdownAnswer.jsx
│   │   └── package.json
│   │
│   └── policypilotai/
│       ├── api.py
│       ├── graph.py
│       ├── guardrails.py
│       ├── ingestion.py
│       ├── llm_gateway.py
│       ├── evaluation.py
│       │
│       └── hybrid_rag/
│           ├── chunking.py
│           ├── keyword_store.py
│           ├── reranker.py
│           ├── retriever.py
│           └── vector_store.py
│
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

## Application Workflow

1. The user submits an HR-related question through the React frontend.
2. FastAPI receives the request and passes it to the LangGraph workflow.
3. Input guardrails validate the question.
4. The retrieval component performs hybrid search using vector similarity and BM25.
5. Relevant document context is passed to the LLM.
6. The LLM generates an answer based on the retrieved context.
7. Output guardrails validate the generated response.
8. The validated answer is streamed to the frontend and displayed to the user.

## Installation and Setup

### Prerequisites

- Python 3.12 or compatible project-supported version
- Node.js and npm
- uv package manager
- Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/VarshaSahu03/PolicyPilotAI.git
cd PolicyPilotAI
```

### 2. Backend Setup

Install Python dependencies:

```bash
uv sync
```

Configure environment variables in a `.env` file using the variable names expected by the application.

For example:

```env
GROQ_API_KEY=your_groq_api_key
```

Ensure the required policy documents and retrieval indexes are prepared using the ingestion pipeline.

Start the FastAPI server:

```bash
uv run uvicorn policypilotai.api:app --app-dir src --reload
```

Backend API documentation:

```text
http://127.0.0.1:8000/docs
```

### 3. Frontend Setup

Open another terminal:

```bash
cd src/frontend
npm install
npm run dev
```

Open the local frontend URL provided by Vite, typically:

```text
http://localhost:5173
```

## Project Objectives

- Simplify access to HR policy information.
- Improve information retrieval using hybrid search.
- Generate context-aware answers grounded in policy documents.
- Improve response reliability through guardrails.
- Provide an interactive conversational interface.

## Author

**Varsha Sahu**

GitHub: [VarshaSahu03](https://github.com/VarshaSahu03)
