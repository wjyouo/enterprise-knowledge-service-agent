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


def load_threads():
    try:
        resp = httpx.get(f"{API}/threads", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.sidebar.error(f"无法加载历史会话：{exc}")
        return []


def load_thread_messages(thread_id: str):
    try:
        resp = httpx.get(f"{API}/threads/{thread_id}/messages", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.sidebar.error(f"无法加载该会话：{exc}")
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
    st.header("🧠 企业知识服务")

    if st.button("➕ 新建会话", use_container_width=True):
        new_chat()

    st.subheader("历史会话")
    threads = load_threads()

    if not threads:
        st.caption("还没有会话。")

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
                st.error(f"删除失败：{exc}")
            st.rerun()

    st.divider()
    st.subheader("📄 导入知识文档")
    uploaded_files = st.file_uploader("上传 PDF 文档", type=["pdf"], accept_multiple_files=True)

    if st.button("导入文档", disabled=not uploaded_files):
        with st.spinner("正在切分、向量化并写入 pgvector..."):
            files_payload = [
                ("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files
            ]
            try:
                resp = httpx.post(f"{API}/ingest", files=files_payload, timeout=120)
                resp.raise_for_status()
                for item in resp.json():
                    st.success(f"{item['filename']}：已写入 {item['chunks_ingested']} 个文本块")
            except Exception as exc:
                st.error(f"导入失败：{exc}")

    st.divider()
    st.caption(f"后端地址：{BACKEND_URL}")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("企业知识服务与工单协同平台")
st.caption("基于 LangGraph 的 RAG 检索、MCP 工具调用、工单协同与 Self-RAG 校验")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            cols = st.columns(2)
            cols[0].caption(f"来源：{', '.join(msg.get('sources', [])) or '无'}")
            cols[1].caption(f"调用工具：{msg.get('tool_used') or '无'}")

question = st.chat_input("请输入你的问题，例如：如何申请 VPN？帮我创建一个数据库权限工单。")

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
                        tool_badge.info(f"🔧 正在调用 `{tool_used}({args_str})`…")

                    elif event == "retry":
                        stage_labels = {
                            "rewriting_query": "✏️ 正在改写检索问题…",
                            "web_search_fallback": "🌐 内部资料不足，正在尝试网页检索…",
                            "revising_answer": "🔁 答案未通过校验，正在重新生成…",
                        }
                        tool_badge.info(stage_labels.get(data.get("stage"), "🔁 正在重试…"))

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
                        answer = f"错误：{data.get('detail', '未知错误')}"
                        answer_area.markdown(answer)
                        tool_badge.empty()

        except Exception as exc:
            answer = f"连接后端失败：{exc}"
            answer_area.markdown(answer)
            tool_badge.empty()

        cols = st.columns(2)
        cols[0].caption(f"来源：{', '.join(sources) or '无'}")
        cols[1].caption(f"调用工具：{tool_used or '无'}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "tool_used": tool_used}
    )
    st.rerun()
