'''
- cloudflare sitemap
    - 예시: https://api.python.langchain.com/sitemap.xml
    - https://developers.cloudflare.com/sitemap-0.xml
    - AIGateway
    - Cloudflare vectorize
    - Workers AI
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



#---------- config
st.set_page_config(
    page_icon="🤣",
    page_title="SiteGPT"
)

#---------- function
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda



def parse_page(soup):        
    return (
        str(soup.get_text()).replace('\n','').replace('\xa0', '')
    )


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",     
        ],
        parsing_function=parse_page
    )
    st.write(loader.load())
    print('loading완료')
    loader.requests_per_second = 2
    docs = loader.load_and_split(
        text_splitter=splitter
    )
    print("분할 시작")   
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    print("vectorstore 완료")
    return vector_store.as_retriever()  

def send_message(role, message):
    with st.chat_message(role):
        st.write(message)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }



def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


#---------- sidebar
with st.sidebar:
    url = st.text_input(
        label="Write down a URL",
        placeholder="https://example.com"
    )


#---------- main
##header
st.title("SiteGPT")

##main
# map re duce 
if url:
    retriever = load_website(url=url)
    st.write('retriever 완료')

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    answers_prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            '''                
                context에 들어오는 데이터를 가지고 아래와 같은 예시로 대답해. 점수는 0점부터 5점까지 줄 수 있어. 답변이 사용자의 질문과 관련이 깊을수록 점수는 높아야 해. 0점이라도 항상 답변의 점수를 포함해

                context: {context}

                Ouput Example:
                    Question: How far away is the moon?
                    Answer: The moon is 384,400 km away.
                    Score: 5
                                                                
                    Question: How far away is the sun?
                    Answer: I don't know
                    Score: 0
            '''
        ),
        ('human','{question}')
    ])

    question = "How many indexes can a single account have in Vectorize?"
    answer_chain = answers_prompt | llm
    answers = []
    for i, doc in enumerate(retriever):    
        print(i)
        print(doc)
        result = answer_chain.invoke(
            {
                'context': doc,
                'question': question
            }
        )
        answers.append(result)
        print(result)
    st.write(answers)


      
    
    
    



    


    # question = "How many indexes can a single account have in Vectorize?"
    # chain = answers_prompt | llm
    # answers = []
    # for doc in retriever:
    #     result = {
    #         'question': question,
    #         'answer': chain.invoke(
    #             {
    #                 'context': doc.page_content,
    #                 'question': question
    #             }
    #         )
    #     }
    #     answers.append(result)
    # st.write(answers)









        