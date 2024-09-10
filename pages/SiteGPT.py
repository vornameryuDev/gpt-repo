'''
- cloudflare sitemap    
    - https://developers.cloudflare.com/sitemap-0.xml
    - filter
        - AIGateway(/ai-gateway/)
        - Cloudflare vectorize(/vectorize/)
        - Workers AI(/workers-ai/)
- Questions
    - "What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?"
    - "What can I do with Cloudflare’s AI Gateway?"
    - "How many indexes can a single account have in Vectorize?"
- conditions
    - Allow the user to use its own OpenAI API Key: st.sidebar, st.text_input
    - put a link to the Github repo: st.sidebar

- 기능구현 순서
    - url > 내용 가져오기 = retriever
        - url입력
        - parsing: url 내용 텍스트로 가져오기
        - split
        - embedding & caching
        - vectorstore
    - {'context': retriever} | example_prompt | llm = chain
    - chain.invoke = result
'''



import streamlit as st
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler


#---------- header & icon
st.set_page_config(
    page_title="SiteGPT",
    page_icon="😍"
)
st.title("Site GTP")


#---------- sidebar
with st.sidebar:
    api_key = st.text_input(
        label="Enter your openAI API-KEY",
        type='password',
    )
    url = st.text_input(
        label="Write down a URL(only sitemap.xml)",
        placeholder="https://example.com/sitemap.xml"
    )
     
            
    # 공간을 많이 띄우기 위해 추가
    st.sidebar.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

    # GitHub 링크를 사이드바 하단에 추가하고 가운데 정렬하기 위한 CSS
    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <a href="https://github.com/vornameryuDev" target="_blank">GitHub 링크</a>
        </div>
        """,
        unsafe_allow_html=True
    )

#---------- main
def parse_page(soup):
    print(str(soup.get_text()))
    return (
        str(soup.get_text()).replace('\n','').replace('\xa0', '')
    )

@st.cache_resource(show_spinner="Scraping & Embedding...")
def load_website(url):
    #load and split
    loader = SitemapLoader(
        web_path=url,
        filter_urls=[
            r"^(.*\/workers-ai\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/ai-gateway\/).*",
        ],
        parsing_function=parse_page
    )
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    print("docs완료")

    #embedding & vectorstore
    embeddings = OpenAIEmbeddings(api_key=api_key)
    print("embedding완료")
    cached_path = LocalFileStore('././.cache/embeddings/sitemap.xml')
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cached_path)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    # vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore.as_retriever()

def get_answers(input):
    docs = input['docs'] #retriever
    question = input['question'] #question        
    answers_chain = answers_prompt | llm
    result = {
        'question': question,
        'answers': [
            {
                'answer': answers_chain.invoke(
                    {'question': question, 'context': doc.page_content}
                ).content,
                'source': doc.metadata['source'],
                "date": doc.metadata['lastmod']
            } for doc in docs
        ]
    }
    print('get_answers완료')
    return result

def choose_answer(input):
    question = input['question'] #question
    answers = input['answers'] #answers
    choose_chain = choose_prompt | llm
    condensed = '\n\n'.join(f"{answer['answer']}\n{answer['source']}" for answer in answers)
    result = choose_chain.invoke(
        {
            'context': condensed,
            'question': question
        } 
    )
    print('choose_answers완료')
    return result

def send_message(role, message, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(role, message)

def paint_history():
    for message in st.session_state['messages']:
        send_message(message['role'], message['message'], save=False)

def save_message(role, message):
    st.session_state['messages'].append(
            {
                'role': role,             
                'message': message,
            }
        )        




answers_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        '''
            주어지는 context를 보고 질문에 대답해. 모르면 꾸며내지말고 모른다고 대답해. 그리고 점수를 부여해. 점수는 0점에서 5점까지야. 질문에 더 적절한 답변일 수록 높은 점수를 부여하고 그렇지 않으면 낮은 점수를 부여해.

            Context: {context}

            Examples:
                Question: How far away is the moon?
                Answer: The moon is 384,400 km away.
                Score: 5
                                                            
                Question: How far away is the sun?
                Answer: I don't know
                Score: 0
        '''
    ),
    ('human', '{question}')
])    
choose_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        """
            질문을 받아 answer, source, date를 추출했고, 이 대답들을 context로 줄거야. answer에는 score가 있어. 대답들 중 score가 높은 답변을 골라. 고른 답변의 출처를 변경하지 말고 그대로 가져와.

            context: {context}
        """
    ),
    ('human', '{question}')
])


if not url:
    st.markdown(
        """
        Ask questions about the content of a website.

        Start by writing the URL of the website on the sidebar.
        """
    )    
    st.session_state['messages'] = []    

elif not ".xml" in url:
    st.error('please write down a Sitemap URL')
elif url:

    retriever = load_website(url)

    send_message('ai', 'Ask about sitemap!', save=False)

    paint_history() #그리기

    llm = ChatOpenAI(
        api_key=api_key,
        temperature=0.1,
        model='gpt-4o-mini'
    )

    #질문
    query = st.chat_input()
    if query:
        send_message('human', query, save=True) # 질문 그리기 > 저장
        chain = (
            {
                'docs': retriever,
                'question': RunnablePassthrough()
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )        
        result = chain.invoke(query)
        send_message('ai', result.content, save=True) #답변 그리기 > 저장
            


    

    

    



    