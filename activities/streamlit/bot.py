from openai import OpenAI
import pandas as pd
import streamlit as st

st.title("TinyLLM Chatbot Local LLM")
st.caption("This is a chatbot powered by TinyLLM running locally on your machine.")

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1", 
    api_key = "sk-no-key-required"
)

def get_response(user_prompt):
    completion = client.chat.completions.create(
    model="TinyLLM",
    messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_prompt}
  ]
)
    return completion.choices[0].message.content


prompt = st.chat_input(
    placeholder="ask me anything...",
)

if prompt:
    response = get_response(prompt)
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(response)
    

