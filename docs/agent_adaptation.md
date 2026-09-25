# IT Knowledge Service And Ticket Agent Adaptation

This fork adapts Enterprise Knowledge Copilot into an internal IT knowledge
service and ticket-collaboration agent.

Verified baseline from the source tree:

- The backend is FastAPI.
- The workflow is built with async LangGraph nodes.
- RAG uses PostgreSQL plus pgvector.
- Business tools are exposed through FastMCP.
- Tickets are stored in the MCP SQLite database.

Adaptation scope started here:

- Add optional Jev typed decisions for high-frequency intent routing.
- Use Jev Choice for route selection and ticket action selection.
- Use Jev Score for support urgency.
- Use Jev Noul for the human-review gate.
- Keep existing LLM routing and deterministic stemming as fallbacks when Jev
  is not configured or returns a low-confidence decision.

Jev configuration:

```env
JEV_API_KEY=your_key
JEV_API_URL=https://api.typesafe.ai/v1/systemone
JEV_MODEL=jev-latest
JEV_MIN_CONFIDENCE=0.62
JEV_HUMAN_REVIEW_THRESHOLD=0.65
```

The current integration does not require a key for local development. Without
JEV_API_KEY, the original LLM and deterministic routers continue to run.
