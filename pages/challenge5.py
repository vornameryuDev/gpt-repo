import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler




#---------- function
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

@st.cache_resource(show_spinner="embedding..")
def embed_file(file, api_key):
    file_name = file.name
    file_path = f'././.cache/files/{file_name}'
    file_content = file.read()
    with open(file_path, "wb") as f:
        f.write(file_content)    

    #load_file    
    loader = TextLoader(file_path)

    #split file
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    print(len(docs))

    #embeddings & cache
    embeddings = OpenAIEmbeddings(api_key=api_key)    
    cache_path = LocalFileStore(f"././.cache/embeddings/{file_name}") 
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_path)

    #vectorstore > retriever
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    print("retriever 생성 완료")
    return retriever

def send_message(role, message, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(role, message)

def paint_history():
    for message in st.session_state['messages']:
        send_message(
            message["role"],
            message["message"],
            save=False,
        )

def save_message(role, message):
    st.session_state["messages"].append({"message": message, "role": role})

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message("ai", self.message)

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)



#---------- sidebar
with st.sidebar:
    api_key = st.text_input(label="Write your API Key")
    file = st.file_uploader(
        label="Upload File",
        type=["txt"]
    )
    st.session_state

#---------- main
if file:
    retriever = embed_file(file, api_key)

    with st.status(label="Loading Document...") as status:
        status.update(label="Embedding Document...")
        status.update(label="Ask about Document!")
    send_message('ai', f'Hello, Ask a question about the {file.name}!!', save=False)

    paint_history()

    message = st.chat_input(placeholder="Ask me about your document")
    if message:
        send_message('human', message, save=True) #질문 그리기 및 저장
        llm = ChatOpenAI(
            temperature=0.1,
            api_key=api_key,
            streaming=True,
            callbacks=[ChatCallbackHandler()]
        )
        prompt = ChatPromptTemplate.from_messages([
            (
                'system',
                """
                    너는 도움을 주는 훌륭한 조수야. 주어지는 문서에서 묻는 말에 대한 답을 찾아줘. 만약 문서안에 답이 없다면 모르겠다고 대답해.\n\n{context}
                """,
            ),
            ('human', '{question}')
        ])
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state['messages'] = []