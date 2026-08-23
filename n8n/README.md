# /n8n — Webhook Front Door & Retry Pipeline

Contains exported n8n workflow definitions (`satquery_workflow.json`):

- **Webhook Node**: Receives incoming requests (`POST /satquery`) from the Next.js frontend.
- **HTTP Node**: Dispatches requests to the LangGraph Orchestrator (`POST /query`).
- **Retry Logic**: Automatically retries 5xx server errors once before returning failure.
- **Pass-through**: Delivers final JSON response directly to client.
