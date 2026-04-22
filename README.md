# 🎬 AutoStream AI Agent
### Social-to-Lead Agentic Workflow | ServiceHive × Inflx Internship Assignment

A Conversational AI Agent for **AutoStream** — an automated video editing SaaS for content creators. The agent detects user intent, answers product questions using RAG, and captures qualified leads through a structured collection flow.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph (StateGraph) |
| LLM | Groq API — `llama-3.1-8b-instant` |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` (local, no API needed) |
| Vector Store | FAISS (local) |
| Language | Python 3.11 |
| Environment | python-dotenv |

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/priyanshu812/autostream-agent.git
cd autostream-agent
```

### 2. Create a Python 3.11 Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First run will download the `all-MiniLM-L6-v2` model (~90MB) from HuggingFace automatically.

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at: https://console.groq.com

### 5. Run the Agent

```bash
python agent.py
```

---

## 💬 Example Conversation

```
You: Hi!
Assistant: Hey! I'm the AutoStream assistant 👋 How can I help you today?

You: What's included in the Pro plan?
Assistant: The Pro plan is $79/month and includes unlimited videos, 4K resolution,
           AI captions, and 24/7 priority support.

You: That sounds great, I want to sign up for my YouTube channel.
Assistant: Awesome! Could I get your name first? 😊

You: Priya Sharma
Assistant: Nice to meet you, Priya! What's your email address?

You: priya@gmail.com
Assistant: Got it! Which platform do you primarily create content on?

You: YouTube
Assistant: You're all set, Priya! 🎉 Welcome to AutoStream!

==================================================
✅ LEAD CAPTURED SUCCESSFULLY
==================================================
  Name     : Priya Sharma
  Email    : priya@gmail.com
  Platform : YouTube
==================================================
```

---

## 🏗️ Architecture Explanation

### Why LangGraph?

LangGraph was chosen for its **explicit, deterministic state management** via a typed `StateGraph`. Every conversation turn passes through a defined pipeline — `detect_intent → router → handler` — making the agent's behavior fully predictable and debuggable. Unlike AutoGen's multi-agent loops, LangGraph gives precise control over when tools fire, which is critical here: `mock_lead_capture()` must only trigger after all three fields (name, email, platform) are collected — never prematurely.

### How State is Managed

The agent uses a `TypedDict`-based `AgentState` that persists across every graph invocation:

- **`messages`** — Full conversation history, append-only via LangGraph's `add_messages` reducer
- **`intent`** — Detected intent string, updated each turn by the `detect_intent` node
- **`collecting_lead`, `collection_step`** — Tracks where we are in the lead collection flow (`name → email → platform → done`)
- **`lead_name`, `lead_email`, `lead_platform`** — Collected incrementally across turns
- **`lead_captured`** — Boolean flag to prevent duplicate lead capture

The `router` function checks `collecting_lead` before intent, ensuring mid-collection messages are never misrouted — even if the user types something that looks like a greeting during the name/email/platform steps.

### RAG Pipeline

The knowledge base (`knowledge_base.md`) is chunked into 500-character segments with 50-character overlap using `RecursiveCharacterTextSplitter`, then embedded locally using HuggingFace's `all-MiniLM-L6-v2` model and stored in a FAISS vectorstore. On each product query, the top-3 most relevant chunks are retrieved and injected into the system prompt as context — no external embedding API needed.

### Why Groq + Local HuggingFace?

Originally built with Gemini 2.0 Flash (LLM) and Gemini Embedding-001 (embeddings). Switched to Groq + HuggingFace for two reasons:
1. **Groq** offers a generous free tier with no daily quota exhaustion issues
2. **Local HuggingFace embeddings** eliminate any API dependency for the RAG pipeline — the vectorstore builds entirely offline after the one-time model download

---

## 📁 Project Structure

```
autostream-agent/
├── agent.py            # LangGraph agent — intent detection, routing, response handlers
├── rag_pipeline.py     # FAISS vectorstore + HuggingFace embeddings
├── tools.py            # mock_lead_capture() tool
├── knowledge_base.md   # AutoStream pricing, features, policies
├── requirements.txt    # All dependencies
├── .env                # GROQ_API_KEY (not committed to Git)
└── README.md           # This file
```

---

## 🔄 Agent Flow

```
User Input
    │
    ▼
detect_intent (Groq LLM)
    │
    ▼
Router
    ├── "greeting"       → handle_greeting      → Warm response        → END
    ├── "product_query"  → handle_product_query  → RAG + Groq response  → END
    └── "high_intent"    → handle_high_intent    → Lead collection      → END
              │
              └── collecting_lead = True?
                   → Always route here until lead_captured = True
                        Step 1: Ask for name
                        Step 2: Ask for email
                        Step 3: Ask for platform
                        Step 4: Fire mock_lead_capture() ✅
```

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the following approach would be used:

### Step 1: WhatsApp Business API Setup
Register on **Meta for Developers**, create a WhatsApp Business App, and obtain a phone number, `WHATSAPP_TOKEN`, and `VERIFY_TOKEN`.

### Step 2: Build a Webhook Server (FastAPI)

```python
from fastapi import FastAPI, Request
app = FastAPI()

@app.get("/webhook")
async def verify(hub_mode, hub_verify_token, hub_challenge):
    if hub_verify_token == "YOUR_VERIFY_TOKEN":
        return int(hub_challenge)

@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    user_phone = data["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    user_text  = data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

    # Retrieve or create session state for this phone number
    state = session_store.get(user_phone, default_state())
    state["messages"].append(HumanMessage(content=user_text))

    # Run the LangGraph agent
    state = app_graph.invoke(state)
    session_store[user_phone] = state

    # Send reply back via WhatsApp Cloud API
    send_whatsapp_message(user_phone, get_last_ai_message(state))
```

### Step 3: Per-User Session Store
Each WhatsApp phone number gets its own `AgentState` stored in a dictionary (or Redis for production). This preserves multi-turn conversation state per user — the same pattern already used in this agent.

### Step 4: Deploy
Host on **Railway**, **Render**, or **AWS Lambda** with a public HTTPS URL, then register it as the webhook endpoint in the Meta Developer dashboard.

---

## 👤 Author

**Priyanshu Soni**
B.Tech CSE (AI/ML) | Jain University, Bangalore
GitHub: [github.com/priyanshu812](https://github.com/priyanshu812)
LinkedIn: [linkedin.com/in/priyanshu-soni-ai](https://linkedin.com/in/priyanshu-soni-ai)
