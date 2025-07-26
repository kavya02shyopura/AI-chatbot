import streamlit as st
import requests
import os
import json

# Set your Gemini API key here or use an environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB-LCoG77IW9KfzEQE3p4wzSZBVCvirwwI")

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

# Custom CSS for modern look
st.markdown(
    """
    <style>
    .stChatMessage {
        background: #f0f2f6;
        border-radius: 1.2em;
        padding: 1em;
        margin-bottom: 1em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .user-msg {
        background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%);
        color: #222;
        text-align: right;
    }
    .ai-msg {
        background: linear-gradient(90deg, #fbc2eb 0%, #a6c1ee 100%);
        color: #222;
        text-align: left;
    }
    .stTextInput > div > div > input {
        border-radius: 1em;
        border: 1px solid #a1c4fd;
        padding: 0.75em;
    }
    .stButton > button {
        border-radius: 1em;
        background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%);
        color: #222;
        font-weight: bold;
        border: none;
        padding: 0.5em 1.5em;
    }
    .error-msg {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%);
        color: #d32f2f;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 My AI Chatbot")
st.caption("Ask me anything! Powered by Gemini API.")

# Check if API key is properly set
if GEMINI_API_KEY in ["your_gemini_api_key_here", ""]:
    st.error("⚠️ Please set your Gemini API key! Add it to your environment variables as GEMINI_API_KEY or update the code.")
    st.info("Get your API key from your Gemini API provider")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages
for msg in st.session_state["messages"]:
    role = msg.get("role", "")
    content = msg.get("content", "")
    css_class = "user-msg" if role == "user" else "ai-msg"
    if role == "assistant" and isinstance(content, str) and content.startswith("Error:"):
        css_class = "error-msg"
    st.markdown(f'<div class="stChatMessage {css_class}"><b>{"You" if role=="user" else "AI"}:</b> {content}</div>', unsafe_allow_html=True)

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message...", "", key="input")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Call Gemini API
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": user_input
                    }
                ]
            }
        ]
    }
    ai_response = None
    try:
        with st.spinner("AI is thinking..."):
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                headers=headers,
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                # Extract generated text from response
                ai_response = ""
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    # Extract text from parts if available
                    if "content" in candidate:
                        content = candidate["content"]
                        if isinstance(content, dict) and "parts" in content:
                            parts = content["parts"]
                            if isinstance(parts, list):
                                ai_response = ""
                                for part in parts:
                                    if "text" in part:
                                        ai_response += part["text"]
                        elif isinstance(content, str):
                            ai_response = content
                        else:
                            ai_response = str(content)
                    else:
                        ai_response = str(candidate)
                if not ai_response:
                    ai_response = "Error: No response content received from Gemini API."
            else:
                ai_response = f"Error: API returned status code {response.status_code} - {response.text}"
    except Exception as e:
        ai_response = f"Error: {str(e)}"

    st.session_state["messages"].append({"role": "assistant", "content": ai_response})
    st.rerun()

# Clear chat button
if st.button("Clear Chat"):
    st.session_state["messages"] = []
    st.rerun()

# Sidebar with persona selector and chat stats
with st.sidebar:
    st.header("🤖 Chatbot Settings")
    persona = st.selectbox(
        "Choose Chatbot Persona:",
        ["Friendly", "Professional", "Creative", "Concise"],
        index=0,
        key="persona_selector"
    )
    st.session_state["persona"] = persona
    st.markdown(f"**Current Persona:** {persona}")
    st.markdown(f"**Messages in this chat:** {len(st.session_state['messages'])}")
    st.markdown("---")
    st.markdown("""
    <small>Powered by Gemini API<br>Made with ❤️ using Streamlit</small>
    """, unsafe_allow_html=True)

# Download chat history as text
if st.session_state["messages"]:
    chat_text = "\n".join([
        f"You: {m['content']}" if m['role'] == 'user' else f"AI: {m['content']}" for m in st.session_state["messages"]
    ])
    st.download_button("Download Chat", chat_text, file_name="chat_history.txt")

# Regenerate last AI response button
if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "assistant":
    if st.button("Regenerate Last Response"):
        # Remove last AI response and rerun the last user message
        last_user_msg = None
        for msg in reversed(st.session_state["messages"][:-1]):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        if last_user_msg:
            st.session_state["messages"] = [m for m in st.session_state["messages"] if m["role"] != "assistant" or m != st.session_state["messages"][-1]]
            # Re-run the last user message as if just submitted
            user_input = last_user_msg
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": GEMINI_API_KEY
            }
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": user_input
                            }
                        ]
                    }
                ]
            }
            ai_response = None
            try:
                with st.spinner("AI is thinking..."):
                    response = requests.post(
                        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    if response.status_code == 200:
                        data = response.json()
                        ai_response = ""
                        if "candidates" in data and len(data["candidates"]) > 0:
                            candidate = data["candidates"][0]
                            if "content" in candidate:
                                content = candidate["content"]
                                if isinstance(content, dict) and "parts" in content:
                                    parts = content["parts"]
                                    if isinstance(parts, list):
                                        ai_response = ""
                                        for part in parts:
                                            if "text" in part:
                                                ai_response += part["text"]
                                elif isinstance(content, str):
                                    ai_response = content
                                else:
                                    ai_response = str(content)
                            else:
                                ai_response = str(candidate)
                        if not ai_response:
                            ai_response = "Error: No response content received from Gemini API."
                    else:
                        ai_response = f"Error: API returned status code {response.status_code} - {response.text}"
            except Exception as e:
                ai_response = f"Error: {str(e)}"
            st.session_state["messages"].append({"role": "assistant", "content": ai_response})
            st.rerun()
