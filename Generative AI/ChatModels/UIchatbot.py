from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodBot",
    page_icon="🤖",
    layout="centered",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 780px;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #c084fc 0%, #818cf8 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    font-weight: 400;
    letter-spacing: 0.02em;
}

/* ── Mode selector card row ── */
.mode-row {
    display: flex;
    gap: 0.75rem;
    margin: 1.5rem 0 2rem;
    justify-content: center;
}
.mode-card {
    flex: 1;
    border: 1.5px solid #1e1e2e;
    border-radius: 14px;
    padding: 1.1rem 0.75rem 0.9rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.18s ease;
    background: #111118;
    color: #9ca3af;
    font-size: 0.85rem;
    font-weight: 500;
    max-width: 180px;
}
.mode-card:hover {
    border-color: #374151;
    color: #e8e8f0;
}
.mode-card.active-funny {
    border-color: #fbbf24;
    background: #1a1608;
    color: #fcd34d;
    box-shadow: 0 0 20px rgba(251,191,36,0.15);
}
.mode-card.active-sad {
    border-color: #60a5fa;
    background: #080e1a;
    color: #93c5fd;
    box-shadow: 0 0 20px rgba(96,165,250,0.15);
}
.mode-card.active-angry {
    border-color: #f87171;
    background: #1a0808;
    color: #fca5a5;
    box-shadow: 0 0 20px rgba(248,113,113,0.15);
}
.mode-icon { font-size: 1.6rem; margin-bottom: 0.35rem; }
.mode-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Chat container ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.25rem 0 1rem;
}

/* ── Message bubbles ── */
.msg-row {
    display: flex;
    align-items: flex-end;
    gap: 0.55rem;
}
.msg-row.user { flex-direction: row-reverse; }

.avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.avatar-bot  { background: #1e1b4b; }
.avatar-user { background: #1a1a2e; }

.bubble {
    max-width: 78%;
    padding: 0.75rem 1rem;
    border-radius: 18px;
    line-height: 1.6;
    font-size: 0.93rem;
    font-family: 'Space Grotesk', sans-serif;
}
.bubble-user {
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    color: #e2e8f0;
    border-bottom-right-radius: 4px;
}
.bubble-bot {
    background: #111118;
    border: 1px solid #1e1e2e;
    color: #d1d5db;
    border-bottom-left-radius: 4px;
}
.bubble-bot.funny  { border-color: #78350f; }
.bubble-bot.sad    { border-color: #1e3a5f; }
.bubble-bot.angry  { border-color: #7f1d1d; }

/* ── Timestamp / mode tag ── */
.msg-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #374151;
    margin-bottom: 0.2rem;
}
.msg-row.user .msg-tag { text-align: right; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #374151;
}
.empty-state .es-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-state p { font-size: 0.9rem; line-height: 1.6; }

/* ── Input area ── */
.input-wrap {
    position: sticky;
    bottom: 0;
    background: linear-gradient(to top, #0a0a0f 80%, transparent);
    padding-top: 1.5rem;
    padding-bottom: 0.5rem;
}

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input {
    background: #111118 !important;
    border: 1.5px solid #1e1e2e !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    caret-color: #818cf8;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
.stTextInput > div > div > input::placeholder { color: #374151 !important; }

div[data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.15s ease !important;
    height: 48px !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #1a1a28;
    margin: 1.25rem 0;
}

/* ── Mode active badge ── */
.active-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.badge-funny { background: #1a1608; color: #fbbf24; border: 1px solid #78350f; }
.badge-sad   { background: #080e1a; color: #60a5fa; border: 1px solid #1e3a5f; }
.badge-angry { background: #1a0808; color: #f87171; border: 1px solid #7f1d1d; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e1e2e; border-radius: 2px; }

/* ── Thinking indicator ── */
.thinking {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: #4b5563;
    font-size: 0.82rem;
    padding: 0.5rem 0;
}
.dot { width: 5px; height: 5px; border-radius: 50%; background: #4b5563;
       animation: blink 1.4s infinite both; }
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%,80%,100%{opacity:0.2} 40%{opacity:1} }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # LangChain message history
if "mode" not in st.session_state:
    st.session_state.mode = None            # "funny" | "sad" | "angry"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # display-only list of dicts


# ── Model (cached) ────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2506", temperature=0.7)

model = get_model()


# ── Mode config ───────────────────────────────────────────────────────────────
MODE_CONFIG = {
    "funny": {
        "icon": "😂",
        "label": "Funny",
        "system": "You are a funny AI agent. You respond to all prompts with humor and wit.",
        "badge_class": "badge-funny",
        "bubble_class": "funny",
        "accent": "#fbbf24",
    },
    "sad": {
        "icon": "😢",
        "label": "Sad",
        "system": "You are a sad AI agent. You respond to all prompts with empathy and understanding.",
        "badge_class": "badge-sad",
        "bubble_class": "sad",
        "accent": "#60a5fa",
    },
    "angry": {
        "icon": "😤",
        "label": "Angry",
        "system": "You are an angry AI agent. You respond to all prompts with frustration and irritation.",
        "badge_class": "badge-angry",
        "bubble_class": "angry",
        "accent": "#f87171",
    },
}


def set_mode(m: str):
    if st.session_state.mode != m:
        st.session_state.mode = m
        sys_msg = MODE_CONFIG[m]["system"]
        st.session_state.messages = [SystemMessage(content=sys_msg)]
        st.session_state.chat_history = []


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">MoodBot</div>
  <div class="hero-sub">Pick a personality. Start talking.</div>
</div>
""", unsafe_allow_html=True)


# ── Mode selector (3 columns) ─────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

for col, key in zip([col1, col2, col3], ["funny", "sad", "angry"]):
    cfg = MODE_CONFIG[key]
    is_active = st.session_state.mode == key
    active_cls = f"active-{key}" if is_active else ""
    with col:
        st.markdown(f"""
        <div class="mode-card {active_cls}">
          <div class="mode-icon">{cfg['icon']}</div>
          <div class="mode-label">{cfg['label']}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"{'✓ Active' if is_active else 'Select'}", key=f"btn_{key}",
                     use_container_width=True):
            set_mode(key)
            st.rerun()

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Active badge + chat area ──────────────────────────────────────────────────
if st.session_state.mode is None:
    st.markdown("""
    <div class="empty-state">
      <div class="es-icon">☝️</div>
      <p>Choose a personality above<br>to begin your conversation.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    cfg = MODE_CONFIG[st.session_state.mode]

    # Badge
    st.markdown(f"""
    <div class="active-badge {cfg['badge_class']}">
      {cfg['icon']} {cfg['label']} Mode Active
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state" style="padding:1.5rem 1rem;">
          <div class="es-icon">💬</div>
          <p>No messages yet — say something!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for entry in st.session_state.chat_history:
            if entry["role"] == "user":
                st.markdown(f"""
                <div class="msg-row user">
                  <div>
                    <div class="msg-tag">You</div>
                    <div class="bubble bubble-user">{entry['content']}</div>
                  </div>
                  <div class="avatar avatar-user">🧑</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-row bot">
                  <div class="avatar avatar-bot">{cfg['icon']}</div>
                  <div>
                    <div class="msg-tag">MoodBot · {cfg['label']}</div>
                    <div class="bubble bubble-bot {cfg['bubble_class']}">{entry['content']}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="input-wrap">', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        inp_col, btn_col = st.columns([5, 1])
        with inp_col:
            user_input = st.text_input(
                "message",
                placeholder=f"Talk to the {cfg['label'].lower()} bot…",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        # Append to display history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Append to LangChain message list
        st.session_state.messages.append(HumanMessage(content=user_input))

        # Call model
        with st.spinner(""):
            response = model.invoke(st.session_state.messages)

        bot_reply = response.content
        st.session_state.messages.append(AIMessage(content=bot_reply))
        st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑 Clear conversation", use_container_width=False):
            sys_msg = cfg["system"]
            st.session_state.messages = [SystemMessage(content=sys_msg)]
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)