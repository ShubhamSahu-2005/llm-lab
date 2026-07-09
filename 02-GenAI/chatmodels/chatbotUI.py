import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Mood AI Chatbot")
st.title("Choose your AI mode")

mode_choice = st.radio(
    "Select a mode:",
    ["1. Angry", "2. Sad", "3. Funny"],
)

mode_map = {
    "1. Angry": "You are an angry AI agent. Respond aggressively and impatiently.",
    "2. Sad": "You are a very sad AI agent. Respond emotionally and depressingly.",
    "3. Funny": "You are a funny and witty AI agent. Respond humorously and sarcastically.",
}

mode = mode_map.get(mode_choice, "You are a polite and neutral AI assistant.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=512,
)

# Reset chat history if mode changes
if "current_mode" not in st.session_state or st.session_state.current_mode != mode:
    st.session_state.current_mode = mode
    st.session_state.messages = [SystemMessage(content=mode)]

# Display chat history (skip system message)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

prompt = st.chat_input("You:")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    response = llm.invoke(st.session_state.messages)
    st.session_state.messages.append(AIMessage(content=response.content))

    with st.chat_message("assistant"):
        st.write(response.content)