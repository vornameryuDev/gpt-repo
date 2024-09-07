import streamlit as st


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough



def embedding_file(file):
    file_name = file.name
    file_content = file.read()
    file_path = f"././.cache/files/{file_name}"
    # file_path = f"/.cache/files/{file_name}"
    with open(file_path, 'wb') as f: 
        f.write(file_content)

    #document_load
    loader = TextLoader(file_path)

    #split
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    
    #embedding
    embeddings = OpenAIEmbeddings()
    cache_dir = LocalFileStore(f"././.cache/embeddings/{file_name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstores = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstores.as_retriever()
    return retriever


def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


def send_message(role, message):
    with st.chat_message(role):
        st.write(message)


#---------- header & config
st.set_page_config(
    page_title="Document Upload Chatbot"
)


st.title('Chatbot for Document')
st.markdown(
    '''
    Upload your document to the sidebar and ask questions.
    
    We'll find information and answer your questions in the documentation.
'''
)


#---------- variables
llm = ChatOpenAI(temperature=0.1)
prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        '''
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
        Context: {context}
        '''
    ),
    ('human', '{question}')
])


#---------- sidebar
with st.sidebar:
    file = st.file_uploader(
        label="Upload File",
        type=['txt']        
    )
    st.session_state


#---------- main
if file:
    #준비
    retriever = embedding_file(file)
    with st.status(label='Loading Documents...') as status:        
        status.update(label='Ask about Document!!')
    send_message('ai', f'Hello, Ask a question about the {file.name}!!')    
    
    #채팅창 입력
    message = st.chat_input(
        placeholder="Ask anyting about your file"
    )
    if message:
        #물어봄
        send_message('human', message)    
        
        #찾아서 답주기
        chain = {
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        } | prompt | llm
        
        result = chain.invoke() #답추출

        #답그리기
        send_message('ai', result)