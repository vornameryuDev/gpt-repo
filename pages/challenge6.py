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
- 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다. [o]: st.warning
- 만점이면 st.ballons를 사용합니다. [o]: st.balloon
- 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다. [o]: text_input

- 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다. [x]
- st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다. [x]

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



#---------- header & title
st.set_page_config(
    page_icon="🎪",
    page_title="QuizGPT"
)

st.title("QuizGPT")
st.markdown(
    '''
        We will create a quiz related to the document you want.
        
        Upload a file or enter a topic.

    '''
)

#---------- function
@st.cache_resource(show_spinner="Searching Wikipedia...")
def wikipedia_search(keyword):    
    retriever = WikipediaRetriever()
    docs = retriever.get_relevant_documents(keyword) # 문서 분할
    return docs


@st.cache_resource(show_spinner="Loading File...")
def split_file(file):
    file_name = file.name
    file_path = f"././.cache/files/{file_name}" #파일 저장경로
    file_content = file.read()
    with open(file_path, 'wb') as f: #파일저장
        f.write(file_content)
    
    loader = TextLoader(file_path=file_path)
    # 문서분할
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

@st.cache_resource(show_spinner="Making quiz...")
def create_quiz(_docs):
    chain = {'context': question_chain} | format_chain | output_parser
    response = chain.invoke(_docs)
    return response

#---------- sidebar
with st.sidebar:
    docs = None
    choice = st.selectbox(
        label="Choice Options",
        options=["File", "Wikipedia"]
    )
    if choice == "File": #파일
        file = st.file_uploader(
            label="Upload File...",
            type=['txt']
        )
        if file:
            docs = split_file(file) #분할 파일
    else: #위키피디아
        keyword = st.text_input(
            label='Enter the keyword you want to search'
        )
        docs = wikipedia_search(keyword=keyword)


#---------- main
llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
question_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        '''        
        You are an assistant in the role of a teacher. Give 4 problems based on the received context. Each problem has 4 options. Only one of the choices is correct. Mark the correct answer using (o).Please refer to the example below.

        Question Examples:
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
                
            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut
                
            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998
                
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model

        Context: {context}
        '''
    )
])
format_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        """
            You are the algorithm that formats the message. It will pass the context. Receive the example input and change it to json format like example output. Answers with (o) are the correct ones. Example input will be passed to context.

            Example Input:
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)
                    
                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut
                    
                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998
                    
                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model

            Example Output:
                ```json
                {{
                    'questions': [
                        {{
                            'question': 'What is the color of the ocean?',
                            'answers': [
                                {{
                                    'answer': 'Red',
                                    'correct': false
                                }},
                                {{
                                    'answer': 'Yellow',
                                    'correct': false
                                }},
                                {{
                                    'answer': 'Green',
                                    'correct': false
                                }},
                                {{
                                    'answer': 'Blue',
                                    'correct': true
                                }}
                            ]
                        }},
                        {{
                            "question": "What is the capital or Georgia?",
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
                            ]
                        }},
                        {{
                            "question": "When was Avatar released?",
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
                            ]
                        }},
                        {{
                            "question": "Who was Julius Caesar?",
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
                            ]
                        }}
                    ]
                }}
                ```
                Context: {context}
            """
    )
])

    
output_parser = JsonOutputParser()

question_chain = {'context': format_docs} | question_prompt | llm
format_chain = format_prompt | llm 



if not docs:
    st.markdown(
        '''
            We will create a quiz related to the document you want.
            
            Upload a file or enter a topic.
        '''
    )
else:
    response = create_quiz(docs) # docs > question > formatting
    st.write(response)
    correct_count = 0 #맞춘 갯수
    total_questions = len(response['questions']) #질문 갯수

    with st.form("questions_form"):
        for question in response["questions"]:            
            answers = [answer['answer'] for answer in question["answers"]]
            value = st.radio(
                "Select an option.",
                answers,
                index=None,
            )
            if {'answer':value, 'correct': True} in question['answers']:
                st.success('Correct!!')
                correct_count += 1 #맞추면 +1
            elif value is not None:
                st.error('Wrong!!!')
        button = st.form_submit_button()
        if button:            
            if correct_count == total_questions: #다맞추면 balloons
                st.balloons()
            else: #못맞추면
                #warning
                st.warning(f"You got {correct_count} out of {total_questions} correct. Please try again!")
        





