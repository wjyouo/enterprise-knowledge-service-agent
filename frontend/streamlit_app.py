"""
Streamlit UI for the Enterprise Knowledge Copilot.

Talks to the async FastAPI backend over HTTP (Streamlit's own execution
model is synchronous-per-rerun, so a plain httpx.Client is used here; all
the actual async work happens server-side in the FastAPI/LangGraph app).

Sidebar shows past chat threads (persisted in Postgres). Selecting one loads
its full message history and continues the same conversation; "New chat"
starts a fresh thread.

Run with:  streamlit run frontend/streamlit_app.py
"""
import json
import os

import httpx
import streamlit as st


TEXT = {
    "zh": {
        "language_label": "语言 / Language",
        "sidebar_title": "🧠 企业知识服务",
        "new_chat": "➕ 新建会话",
        "history": "历史会话",
        "no_conversations": "还没有会话。",
        "load_history_failed": "无法加载历史会话：{error}",
        "load_thread_failed": "无法加载该会话：{error}",
        "delete_failed": "删除失败：{error}",
        "ingest_title": "📄 导入知识文档",
        "upload_pdf": "上传 PDF 文档",
        "ingest_button": "导入文档",
        "ingesting": "正在切分、向量化并写入 pgvector...",
        "ingest_success": "{filename}：已写入 {chunks} 个文本块",
        "ingest_failed": "导入失败：{error}",
        "backend": "后端地址：{url}",
        "main_title": "企业知识服务与工单协同平台",
        "main_caption": "基于 LangGraph 的 RAG 检索、MCP 工具调用、工单协同与 Self-RAG 校验",
        "sources": "来源：{sources}",
        "tool_used": "调用工具：{tool}",
        "none": "无",
        "chat_placeholder": "请输入你的问题，例如：如何申请 VPN？帮我创建一个数据库权限工单。",
        "running_tool": "🔧 正在调用 `{tool}({args})`…",
        "rewriting_query": "✏️ 正在改写检索问题…",
        "web_search_fallback": "🌐 内部资料不足，正在尝试网页检索…",
        "revising_answer": "🔁 答案未通过校验，正在重新生成…",
        "retrying": "🔁 正在重试…",
        "event_error": "错误：{error}",
        "backend_error": "连接后端失败：{error}",
    },
    "en": {
        "language_label": "Language / 语言",
        "sidebar_title": "🧠 Enterprise Knowledge Service",
        "new_chat": "➕ New chat",
        "history": "History",
        "no_conversations": "No conversations yet.",
        "load_history_failed": "Could not load history: {error}",
        "load_thread_failed": "Could not load this thread: {error}",
        "delete_failed": "Delete failed: {error}",
        "ingest_title": "📄 Ingest documents",
        "upload_pdf": "Upload PDF documents",
        "ingest_button": "Ingest",
        "ingesting": "Chunking, embedding, and storing in pgvector...",
        "ingest_success": "{filename}: stored {chunks} chunks",
        "ingest_failed": "Ingestion failed: {error}",
        "backend": "Backend: {url}",
        "main_title": "Enterprise Knowledge Service And Ticket Agent",
        "main_caption": "LangGraph-powered RAG retrieval, MCP tool calls, ticket collaboration, and Self-RAG verification",
        "sources": "Sources: {sources}",
        "tool_used": "Tool used: {tool}",
        "none": "none",
        "chat_placeholder": "Ask a question, for example: How do I request VPN access? Create a database permission ticket.",
        "running_tool": "🔧 Running `{tool}({args})`...",
        "rewriting_query": "✏️ Rewriting the search query...",
        "web_search_fallback": "🌐 Internal context is insufficient, trying web search...",
        "revising_answer": "🔁 The answer did not verify, regenerating...",
        "retrying": "🔁 Retrying...",
        "event_error": "Error: {error}",
        "backend_error": "Error contacting backend: {error}",
    },
}


def t(key: str, **kwargs) -> str:
    lang = st.session_state.get("language", "zh")
    value = TEXT[lang][key]
    return value.format(**kwargs) if kwargs else value


def _iter_sse_events(response: httpx.Response):
    """Parse a text/event-stream response into (event, data) pairs. Each SSE
    frame is 'event: <name>\\ndata: <json>\\n\\n'; this just accumulates
    lines until a blank line closes a frame."""
    event_name = None
    data_lines = []
    for raw_line in response.iter_lines():
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
        if line == "":
            if event_name is not None:
                try:
                    payload = json.loads("\n".join(data_lines)) if data_lines else {}
                except json.JSONDecodeError:
                    payload = {}
                yield event_name, payload
            event_name, data_lines = None, []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())


def _get_backend_url() -> str:
    """Resolve BACKEND_URL from (in order): st.secrets, env var, default.
    st.secrets raises StreamlitSecretNotFoundError if no secrets.toml exists
    at all, so it must be probed inside a try/except rather than via
    st.secrets.get(...), which still triggers the same parse internally."""
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"]
    except Exception:
        pass
    return os.getenv("BACKEND_URL", "http://localhost:8000")


BACKEND_URL = _get_backend_url()
API = f"{BACKEND_URL}/api/v1"

st.set_page_config(page_title="企业知识服务与工单协同平台", page_icon="🧠", layout="wide")

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources", "verification"}]
if "language" not in st.session_state:
    st.session_state.language = "zh"


def load_threads():
    try:
        resp = httpx.get(f"{API}/threads", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.sidebar.error(t("load_history_failed", error=exc))
        return []


def load_thread_messages(thread_id: str):
    try:
        resp = httpx.get(f"{API}/threads/{thread_id}/messages", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.sidebar.error(t("load_thread_failed", error=exc))
        return []


def _tool_from_sources(sources):
    """Sources look like ['external_tools:query_tickets'] since the backend
    now tags which MCP tool ran; pull that back out for display."""
    for s in sources or []:
        if s.startswith("external_tools:"):
            return s.split(":", 1)[1]
    return None


def select_thread(thread_id: str):
    st.session_state.active_thread_id = thread_id
    st.session_state.messages = [
        {
            "role": m["role"],
            "content": m["content"],
            "sources": m.get("sources", []),
            "tool_used": _tool_from_sources(m.get("sources", [])),
        }
        for m in load_thread_messages(thread_id)
    ]


def new_chat():
    st.session_state.active_thread_id = None
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar: chat history + document ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    selected_language = st.selectbox(
        t("language_label"),
        ["中文", "English"],
        index=0 if st.session_state.language == "zh" else 1,
    )
    language = "zh" if selected_language == "中文" else "en"
    if language != st.session_state.language:
        st.session_state.language = language
        st.rerun()

    st.header(t("sidebar_title"))

    if st.button(t("new_chat"), use_container_width=True):
        new_chat()

    st.subheader(t("history"))
    threads = load_threads()

    if not threads:
        st.caption(t("no_conversations"))

    for thread in threads:
        is_active = thread["id"] == st.session_state.active_thread_id
        cols = st.columns([5, 1])
        label = ("🟢 " if is_active else "") + thread["title"]
        if cols[0].button(label, key=f"thread_{thread['id']}", use_container_width=True):
            select_thread(thread["id"])
            st.rerun()
        if cols[1].button("🗑", key=f"del_{thread['id']}"):
            try:
                httpx.delete(f"{API}/threads/{thread['id']}", timeout=30)
                if is_active:
                    new_chat()
            except Exception as exc:
                st.error(t("delete_failed", error=exc))
            st.rerun()

    st.divider()
    st.subheader(t("ingest_title"))
    uploaded_files = st.file_uploader(t("upload_pdf"), type=["pdf"], accept_multiple_files=True)

    if st.button(t("ingest_button"), disabled=not uploaded_files):
        with st.spinner(t("ingesting")):
            files_payload = [
                ("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files
            ]
            try:
                resp = httpx.post(f"{API}/ingest", files=files_payload, timeout=120)
                resp.raise_for_status()
                for item in resp.json():
                    st.success(
                        t(
                            "ingest_success",
                            filename=item["filename"],
                            chunks=item["chunks_ingested"],
                        )
                    )
            except Exception as exc:
                st.error(t("ingest_failed", error=exc))

    st.divider()
    st.caption(t("backend", url=BACKEND_URL))

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title(t("main_title"))
st.caption(t("main_caption"))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            cols = st.columns(2)
            cols[0].caption(t("sources", sources=", ".join(msg.get("sources", [])) or t("none")))
            cols[1].caption(t("tool_used", tool=msg.get("tool_used") or t("none")))

question = st.chat_input(t("chat_placeholder"))

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question, "sources": [], "tool_used": None})

    with st.chat_message("assistant"):
        tool_badge = st.empty()
        answer_area = st.empty()
        answer = ""
        sources = []
        tool_used = None

        try:
            with httpx.stream(
                "POST",
                f"{API}/chat/stream",
                json={"question": question, "thread_id": st.session_state.active_thread_id},
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for event, data in _iter_sse_events(resp):
                    if event == "tool_call":
                        tool_used = data.get("tool_used")
                        args = data.get("arguments") or {}
                        args_str = ", ".join(f"{k}={v}" for k, v in args.items() if v is not None)
                        tool_badge.info(t("running_tool", tool=tool_used, args=args_str))

                    elif event == "retry":
                        stage_labels = {
                            "rewriting_query": t("rewriting_query"),
                            "web_search_fallback": t("web_search_fallback"),
                            "revising_answer": t("revising_answer"),
                        }
                        tool_badge.info(stage_labels.get(data.get("stage"), t("retrying")))

                    elif event == "token":
                        answer += data.get("text", "")
                        answer_area.markdown(answer + "▌")

                    elif event == "done":
                        sources = data.get("sources", [])
                        tool_used = data.get("tool_used", tool_used)
                        st.session_state.active_thread_id = data.get("thread_id")
                        answer_area.markdown(answer)
                        tool_badge.empty()

                    elif event == "error":
                        answer = t("event_error", error=data.get("detail", "unknown error"))
                        answer_area.markdown(answer)
                        tool_badge.empty()

        except Exception as exc:
            answer = t("backend_error", error=exc)
            answer_area.markdown(answer)
            tool_badge.empty()

        cols = st.columns(2)
        cols[0].caption(t("sources", sources=", ".join(sources) or t("none")))
        cols[1].caption(t("tool_used", tool=tool_used or t("none")))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "tool_used": tool_used}
    )
    st.rerun()
