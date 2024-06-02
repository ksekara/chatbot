import streamlit as st
from main import response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("You:")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Generate chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = response(prompt)
            st.markdown(response_text)
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})
