# agent.py
# AutoStream Conversational AI Agent
# Built with LangGraph + Gemini 1.5 Flash + FAISS RAG

import os
from dotenv import load_dotenv
load_dotenv()  # loads GOOGLE_API_KEY from .env file

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

from rag_pipeline import build_vectorstore, retrieve_context
from tools import mock_lead_capture

# ─────────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # Full conversation history
    intent: str                                # Current detected intent
    lead_name: str                             # Collected lead info
    lead_email: str
    lead_platform: str
    lead_captured: bool                        # Whether lead was already saved
    collecting_lead: bool                      # Whether we're in lead collection mode
    collection_step: str                       # "name" | "email" | "platform" | "done"


# ─────────────────────────────────────────────
# 2. LLM SETUP
# ─────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.3
)

# ─────────────────────────────────────────────
# 3. BUILD VECTORSTORE (once at startup)
# ─────────────────────────────────────────────

print("🔧 Building knowledge base vectorstore...")
vectorstore = build_vectorstore("knowledge_base.md")
print("✅ Vectorstore ready.\n")

# ─────────────────────────────────────────────
# 4. INTENT DETECTION NODE
# ─────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an intent classifier for AutoStream, a video editing SaaS.
Classify the user's latest message into EXACTLY one of these intents:

- "greeting"       : Simple hello, hi, greetings, small talk
- "product_query"  : Questions about features, pricing, plans, policies, how it works
- "high_intent"    : User clearly wants to sign up, try, purchase, or start using the product

Reply with ONLY one word: greeting, product_query, or high_intent. Nothing else."""


def detect_intent(state: AgentState) -> AgentState:
    """Classifies user intent from the latest message."""
    last_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=last_message)
    ])

    intent = response.content.strip().lower()

    # Fallback if LLM returns unexpected value
    if intent not in ["greeting", "product_query", "high_intent"]:
        intent = "product_query"

    return {**state, "intent": intent}


# ─────────────────────────────────────────────
# 5. RESPONSE NODES
# ─────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are a friendly and helpful sales assistant for AutoStream — 
an AI-powered video editing SaaS for content creators.

Your job:
- Answer questions clearly and accurately using the provided knowledge base context
- Be conversational and warm, not robotic
- Never make up pricing or features not in the context
- Keep responses concise (3-5 sentences max)

Context from knowledge base:
{context}
"""


def handle_greeting(state: AgentState) -> AgentState:
    """Handles casual greetings."""
    last_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content="""You are a friendly assistant for AutoStream, a video editing SaaS.
Respond warmly to greetings. Briefly introduce yourself and ask how you can help.
Keep it to 2 sentences."""),
        HumanMessage(content=last_message)
    ])

    new_message = AIMessage(content=response.content)
    return {**state, "messages": state["messages"] + [new_message]}


def handle_product_query(state: AgentState) -> AgentState:
    """Handles product/pricing questions using RAG."""
    last_message = state["messages"][-1].content

    # Retrieve relevant context from knowledge base
    context = retrieve_context(vectorstore, last_message)

    system_prompt = AGENT_SYSTEM_PROMPT.format(context=context)

    # Build message history for multi-turn context
    history = []
    for msg in state["messages"][:-1]:  # exclude latest (already in last_message)
        if isinstance(msg, HumanMessage):
            history.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            history.append(AIMessage(content=msg.content))

    response = llm.invoke(
        [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=last_message)]
    )

    new_message = AIMessage(content=response.content)
    return {**state, "messages": state["messages"] + [new_message]}


def handle_high_intent(state: AgentState) -> AgentState:
    """
    Starts or continues the lead collection flow.
    Collects name → email → platform → triggers mock_lead_capture.
    """
    last_message = state["messages"][-1].content

    # If lead already captured, don't re-capture
    if state.get("lead_captured"):
        new_message = AIMessage(content="You're all set! Our team will reach out to you shortly. 🎉")
        return {**state, "messages": state["messages"] + [new_message]}

    # ── COLLECTION FLOW ──

    collecting = state.get("collecting_lead", False)
    step = state.get("collection_step", "name")

    # First time detecting high intent → start collection
    if not collecting:
        new_message = AIMessage(
            content="That's awesome! I'd love to get you started on the Pro plan. "
                    "Could I get your name first? 😊"
        )
        return {
            **state,
            "messages": state["messages"] + [new_message],
            "collecting_lead": True,
            "collection_step": "name"
        }

    # Collecting name
    if step == "name":
        name = last_message.strip()
        new_message = AIMessage(
            content=f"Nice to meet you, {name}! What's your email address?"
        )
        return {
            **state,
            "messages": state["messages"] + [new_message],
            "lead_name": name,
            "collection_step": "email"
        }

    # Collecting email
    if step == "email":
        email = last_message.strip()
        new_message = AIMessage(
            content=f"Got it! And which platform do you primarily create content on? "
                    f"(e.g. YouTube, Instagram, TikTok, etc.)"
        )
        return {
            **state,
            "messages": state["messages"] + [new_message],
            "lead_email": email,
            "collection_step": "platform"
        }

    # Collecting platform → trigger lead capture
    if step == "platform":
        platform = last_message.strip()

        # Call the mock lead capture tool
        result = mock_lead_capture(
            name=state.get("lead_name", ""),
            email=state.get("lead_email", ""),
            platform=platform
        )

        new_message = AIMessage(
            content=f"Perfect! You're all signed up, {state.get('lead_name')}! 🎉\n\n"
                    f"Our team will reach out to your email shortly with next steps. "
                    f"Welcome to AutoStream — can't wait to see what you create on {platform}! 🚀"
        )
        return {
            **state,
            "messages": state["messages"] + [new_message],
            "lead_platform": platform,
            "lead_captured": True,
            "collection_step": "done"
        }

    # Fallback
    new_message = AIMessage(content="Thanks! Is there anything else I can help you with?")
    return {**state, "messages": state["messages"] + [new_message]}


# ─────────────────────────────────────────────
# 6. ROUTER — decides which node to call next
# ─────────────────────────────────────────────

def router(state: AgentState) -> Literal["handle_greeting", "handle_product_query", "handle_high_intent"]:
    """
    Routes to the correct handler based on:
    - If already collecting lead info → always go to high_intent handler
    - Otherwise → route by detected intent
    """
    if state.get("collecting_lead") and not state.get("lead_captured"):
        return "handle_high_intent"

    intent = state.get("intent", "product_query")

    if intent == "greeting":
        return "handle_greeting"
    elif intent == "high_intent":
        return "handle_high_intent"
    else:
        return "handle_product_query"


# ─────────────────────────────────────────────
# 7. BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_product_query", handle_product_query)
    graph.add_node("handle_high_intent", handle_high_intent)

    # Entry point
    graph.set_entry_point("detect_intent")

    # Conditional routing after intent detection
    graph.add_conditional_edges(
        "detect_intent",
        router,
        {
            "handle_greeting": "handle_greeting",
            "handle_product_query": "handle_product_query",
            "handle_high_intent": "handle_high_intent"
        }
    )

    # All handlers end the graph turn
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_product_query", END)
    graph.add_edge("handle_high_intent", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 8. MAIN CHAT LOOP
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🎬 AutoStream AI Assistant")
    print("  Type 'exit' or 'quit' to stop")
    print("=" * 55 + "\n")

    app = build_graph()

    # Initial state
    state = AgentState(
        messages=[],
        intent="",
        lead_name="",
        lead_email="",
        lead_platform="",
        lead_captured=False,
        collecting_lead=False,
        collection_step="name"
    )

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("\nAssistant: Thanks for chatting! Bye! 👋\n")
            break

        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the graph
        state = app.invoke(state)

        # Print assistant's last message
        last_ai_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai_msg = msg.content
                break

        if last_ai_msg:
            print(f"\nAssistant: {last_ai_msg}\n")


if __name__ == "__main__":
    main()
