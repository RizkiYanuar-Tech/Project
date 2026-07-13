import streamlit as st
from src import load_model, rag_pipeline

@st.cache_resource(show_spinner="Load Model")
def model_classifier():
    return load_model()

classifier_model, intent_label = model_classifier()

st.set_page_config(
    page_title="RAG Customer Support",
    layout="wide",
    initial_sidebar_state='collapsed'
)

# MEMORI
# MEMORI OBROLAN LAYAR UTAMA
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.toast("AI siap digunakan!")

# MEMORI HISTORY OBROLAN
if 'history' not in st.session_state:
    st.session_state.history = []

def add_chat():
    if len(st.session_state.messages) > 0:
        st.session_state.history.append(st.session_state.messages.copy())
        st.session_state.messages = []

def delete_chat(index):
    st.session_state.history.pop(index)

# AREA SIDEBAR
with st.sidebar:
    st.title("Chat History")
    st.button(label='Add Chat', icon='➕', on_click=add_chat)
    st.divider()

    if not st.session_state.history:
        st.caption("Belum ada obrolan")
    else:
        st.write("Riwayat Sebelumnya:")
        for idx, chat in enumerate(st.session_state.history):
            title = ''
            for msg in chat:
                if msg['role'] == 'user':
                    text = msg['content']
                    title = text[:15] + '...' if len(text) > 15 else text
                    break
            col1, col2 = st.columns(spec=[4, 2], vertical_alignment='center')

            with col1:
                if st.button(f'{title}', key=f'{idx}'):
                    st.session_state.messages = chat.copy()
                    st.rerun()       
            with col2:
                st.button("🗑", args=(idx,), on_click=delete_chat)

# OBROLAN UTAMA
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# INPUT USER
if query := st.chat_input(placeholder="Ada yang bisa dibantu?"):
    with st.chat_message("user"):
        st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("Assistant"):
        with st.spinner("Mencari Solusi Keluhan"):
            try:
                response = rag_pipeline(query, classifier_model, intent_label)
                st.markdown(response)
                st.session_state.messages.append({"role": "Assistant", "content": response})
            except Exception as e:
                st.error(f"Terjadi kesalahan {e}")
