import streamlit as st
from src import load_model, rag_pipeline

@st.cache_resource(show_spinner="Load Model")
def model_classifier():
    return load_model()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.message = []
    classifier_model, intent_label = model_classifier()
    st.toast("AI siap digunakan")
else:
    classifier_model, intent_label = model_classifier()

st.set_page_config(
    page_title="RAG Customer Support",
    layout="centered",
    initial_sidebar_state='collapsed'
)

with st.sidebar:
    st.title("Chat History")
    st.divider()
    # Display chat from history
    for message in st.session_state.message:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if query := st.chat_input("Ada yang bisa dibantu?"):
    with st.chat_message("user"):
        st.markdown(query)
        st.session_state.message.append({"role": "user", "content": query})

    with st.chat_message("Alexa"):
        with st.spinner("Mencari Solusi Keluhan"):
            try:
                response = rag_pipeline(query, classifier_model, intent_label)
                st.markdown(response)
                st.session_state.message.append({"role": "Alexa", "content": response})
            except Exception as e:
                st.error(f"Terjadi kesalahan {e}")
