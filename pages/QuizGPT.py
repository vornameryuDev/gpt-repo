'''
1. 퀴즈낼 파일 정의[o]
    1) file upload
    2) Wikipedia
        - 키워드 검색
        - 검색내용 Wikipedia에서 찾기

2. 정의된 파일 분할(docs) [o]
3. 문제와 답 만들기[o]
    question: .....,
    answer: ....(o), ....(x), ....(x)

    question: .....,
    answer: ....(o), ....(x), ....(x)
4. 포매팅 하기[o]
    question: .....,
    answers: [
        {answer: ...., correct: True},
        {answer: ...., correct: False},
        {answer: ...., correct: False},
    ]


[과제]
- 함수 호출을 사용 [o]
- 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용[o]: st.warning
- 만점이면 st.ballons를 사용 [o]: st.balloon
- 유저가 자체 OpenAI API 키를 사용하도록 허용, st.sidebar 내부의 st.input에서 로드합니다. [o]: text_input

- 유저가 시험의 난이도를 커스터마이징 할 수 있도록, LLM이 어려운 문제 또는 쉬운 문제를 생성 [o]
- st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크 삽입 [x]

'''



import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
import json
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler





st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


#---------- function
class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"././.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_resource(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_resource(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


#---------- variable
output_parser = JsonOutputParser()
questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an assistant in the role of a teacher. Give 5 problems based on the received context. Each problem has 4 options. Only one of the choices is correct. Mark the correct answer using (o).Please refer to the example below. The difficulty levels of the questions are Hard and Easny. Set it randomly. And please specify the difficulty level next to the problem.
         
                Question examples:                    
                    Question: What is the color of the ocean? (Hard)
                    Answers: Red|Yellow|Green|Blue(o)
                        
                    Question: What is the capital or Georgia? (Easy)
                    Answers: Baku|Tbilisi(o)|Manila|Beirut
                        
                    Question: When was Avatar released? (Easy)
                    Answers: 2007|2001|2009(o)|1998
                        
                    Question: Who was Julius Caesar? (Hard)
                    Answers: A Roman Emperor(o)|Painter|Actor|Model
                                    
                Context: {context}
            """
        )
    ]
)
formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a powerful formatting algorithm.
                
                You format exam questions into JSON format.
                Answers with (o) are the correct ones.
                
                Example Input:
                    Question: What is the color of the ocean? (Hard)
                    Answers: Red|Yellow|Green|Blue(o)
                        
                    Question: What is the capital or Georgia? (Easy)
                    Answers: Baku|Tbilisi(o)|Manila|Beirut
                        
                    Question: When was Avatar released? (Easy)
                    Answers: 2007|2001|2009(o)|1998
                        
                    Question: Who was Julius Caesar? (Hard)
                    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
                Example Output:                
                    ```json
                    {{ "questions": [
                            {{
                                "question": "What is the color of the ocean? (Hard)",
                                "answers": [
                                    {{
                                        "answer": "Red",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Yellow",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Green",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Blue",
                                        "correct": true
                                    }},
                                ],
                                "level": "Hard"
                            }},
                            {{
                                "question": "What is the capital or Georgia? (Easy)",
                                "answers": [
                                    {{
                                        "answer": "Baku",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Tbilisi",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "Manila",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Beirut",
                                        "correct": false
                                    }},
                                ],
                                "level": "Easy"
                            }},
                            {{
                                "question": "When was Avatar released? (Easy)",
                                "answers": [
                                    {{
                                        "answer": "2007",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "2001",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "2009",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "1998",
                                        "correct": false
                                    }},
                                ],
                                "level": "Easy"
                            }},
                            {{
                                "question": "Who was Julius Caesar? (Hard)",
                                "answers": [
                                    {{
                                        "answer": "A Roman Emperor",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "Painter",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Actor",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Model",
                                        "correct": false
                                    }},
                                ],
                                "level": "Hard"
                            }}
                        ]
                    }}
                    ```
                Questions: {context}
            """,
        )
    ]
)





#---------- sidebar
with st.sidebar:
    docs = None
    #apikey    
    api_key = st.text_input(
        label="Enter your openAI API-KEY",
        type='password',
    )
    
    #select box(level)
    level = st.selectbox(
        label="Choice Levels",
        options=("Hard", "Easy", "All")
    )

    #select box(file)
    choice = st.selectbox(
        label="Choice Options",
        options=("File", "Wikipedia")
    )
    
    if choice == "File": #파일 검색 > docs
        file = st.file_uploader(
            label="Upload File...",
            type=['txt']
        )
        if file:
            docs = split_file(file) #분할 파일
    else: #위키피디아 검색 > docs
        keyword = st.text_input(
            label='Enter the keyword you want to search'
        )
        if keyword:
            docs = wiki_search(keyword)
    

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

if not docs:
    st.markdown(
    """
        We will create a quiz related to the document you want.
                
        Upload a file or enter a topic.
    """
    )
else:
    #모델정의
    llm = ChatOpenAI(
        api_key=api_key,
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    
    #체인정의
    questions_chain = {"context": format_docs} | questions_prompt | llm
    formatting_chain = formatting_prompt | llm

    #문제만들기
    response = run_quiz_chain(docs, keyword if keyword else file.name)
    correct_count = 0 #맞춘 갯수
    total_questions = len(response['questions']) #질문 갯수

    with st.form("questions_form"):
        for i, question in enumerate(response["questions"]):
            #설정한 레벨만 나오게 하기
            if question['level'] == level:

                st.write(f"**{i+1}. {question['question']}**")
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=f"q_{i}"
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    correct_count += 1
                elif value is not None:
                    st.error("Wrong!")
                    
        button = st.form_submit_button()
        if button:            
            if correct_count == total_questions: #다맞추면 balloons
                st.balloons()
            else: #못맞추면
                #warning
                st.warning(f"You got {correct_count} out of {total_questions} correct. Please try again!")





