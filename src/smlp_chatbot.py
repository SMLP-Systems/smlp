# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import streamlit as st
import requests


# This module immplements Streamlit app that:
# -- Accepts user input describing a task in natural language
# -- Sends it to smlp_agent_api.py backend (POST /chat)
# -- Displays the resulting dictionary and any execution result
# To run chatbot locally:
# 1. Make sure your FastAPI-based smlp_agent_api.py is running:
#    python -m uvicorn smlp_agent_api:app --reload --port 8000
#    Here:
#       api_smlp_agent = the module
#       app = the FastAPI() object it should expose
#       --reload = auto-reload on code changes
#       --port 8000 = standard API port
# 2. Run the Streamlit UI:
#      --after installing streamlit in SMLP python virtual environment:
#         python -m streamlit run smlp_chatbot.py
#     --or use streamlit in your system, if it is availlable/installed:
#        streamlit run smlp_chatbot.py
# 3. Open the UI in your browser (usually at http://localhost:8501)
#
#
# @app.post("/load_prompt") -- Called only if you explicitly POST to that URL
# FastAPI route dispatch -- Based purely on @app.method("/path") and incoming HTTP request
#

API_URL = "http://localhost:8000/chat"  # Update if your API runs on a different port or host

st.set_page_config(page_title="SMLP Chatbot", layout="wide")
st.title("🧠 SMLP Assistant")
st.write("Enter a natural language description of the SMLP task you'd like to run.")

# setup a sidebar to enable accessing few-shot promt file
st.sidebar.header("🧠 Few-shot Prompt Loader")
uploaded_file = st.sidebar.file_uploader("Upload few-shot prompt (.txt)", type=["txt"])
if uploaded_file:
    prompt_text = uploaded_file.read().decode("utf-8")
    response = requests.post("http://localhost:8000/load_prompt", json={"prompt": prompt_text})
    if response.ok and response.json().get("status") == "ok":
        st.sidebar.success("Few-shot prompt loaded!")
        st.sidebar.code(prompt_text[:500], language="markdown")

    else:
        st.sidebar.error(f"Error: {response.json().get('message')}")

prompt_path = st.sidebar.text_input("Enter path to few-shot prompt (optional):")
if st.sidebar.button("Load prompt from path") and prompt_path:
    try:
        with open(prompt_path, "r") as f:
            prompt_text = f.read()
        response = requests.post("http://localhost:8000/load_prompt", json={"prompt": prompt_text})
        if response.ok and response.json().get("status") == "ok":
            st.sidebar.success(f"Loaded prompt from: {prompt_path}")
            st.sidebar.code(prompt_text[:500], language="markdown")
        else:
            st.sidebar.error(f"Error loading prompt: {response.json().get('message')}")
    except Exception as e:
        st.sidebar.error(f"File read error: {e}")

# Initialize session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Your instruction:", height=120)
    submitted = st.form_submit_button("Run SMLP")

if submitted and user_input:
    st.session_state.messages.append(("user", user_input))

    # Send request to SMLP agent API
    try:
        response = requests.post(API_URL, json={"message": user_input})
        response.raise_for_status()
        result = response.json()
        agent_output = result.get("result", result)  # fallback to full if no result key
        st.session_state.messages.append(("agent", agent_output))
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.session_state.messages.append(("agent", error_msg))

# Display conversation
for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**SMLP Agent:**\n```json\n{msg}\n```")
