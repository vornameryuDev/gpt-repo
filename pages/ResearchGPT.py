from time import time
import streamlit as st



#---------- header & config

st.set_page_config(
    page_icon="📂",
    page_title="ResearchGPT"
)

st.title('ResearchGPT')


#---------- sidebar
with st.sidebar:
    api_key = st.text_input(
        label="Enter your openAI API-KEY",
        type='password',
    )
    if api_key:
        st.success('Entered API-KEY!!')
        


#---------- main
import wikipediaapi
import requests
from bs4 import BeautifulSoup as bs
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain_community.document_loaders import WebBaseLoader
import json
from openai import OpenAI
import time



def QueryResearchUrl(input):
    query = input['query']
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    url = list(wikipedia.run(query))
    return wikipedia.run(query)


def UrlContentScrappingTool(input):
    url = input['url']
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[0].page_content.replace('\n','')


def SaveToFileTool(input):
    content = input['content']
    filename='research2.txt'
    with open(f'./files/{filename}', 'w', encoding='utf-8') as f:
        f.write(content)
    return f"{filename}에 저장 완료"


functions = [
    {
        'type': 'function',
        'function': {
            'name': 'QueryResearchUrl',
            'description': 'Tool to find Wikipedia URL for a query',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'What comes in when you invoke the agent'
                    }
                },
                'required': ['query']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'UrlContentScrappingTool',
            'description': 'A tool that accesses URLs and scrapes content',
            'parameters': {
                'type': 'object',
                'properties': {
                    'url': {
                        'type': 'string',
                        'description': 'URL from QueryResearchUrlTool'
                    }
                },
                'required': ['url']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'SaveToFileTool',
            'description': 'Save the content returned with UrlContentScrappingTool as a file.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'content': {
                        'type': 'string',
                        'description': 'Content to be saved to the text file'
                    }
                },
                'required': ['content']
            }
        }
    }
]

functions_name = {
    'QueryResearchUrl': QueryResearchUrl,
    'UrlContentScrappingTool': UrlContentScrappingTool,
    'SaveToFileTool': SaveToFileTool
}

@st.cache_resource(show_spinner="Create assistant...")
def get_assistant(model, functions):
    assistant = client.beta.assistants.create(
        name="research assistant",
        instructions="You are a capable assistant who helps me find what I want. When you receive a query, find the URL in Wikipedia and scrape the content. After scraping, save it as a text file.",
        tools=functions,
        model=model
    )
    return assistant


def get_thread(role, content):
    thread = client.beta.threads.create(
        messages=[
            {
                'role': role,
                'content': content
            }
        ]
    )
    return thread

def get_run(assistant_id, thread_id):
    run = client.beta.threads.runs.create(
            assistant_id=assistant_id,
            thread_id=thread_id
    )
    return run

def get_run_retrieve(run_id, thread_id):
    run_retrieve = client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id
    )
    return run_retrieve

def submit_tool_outputs(run_id, thread_id, run_retrieve):
    required_actions = run_retrieve.required_action.submit_tool_outputs.tool_calls

    client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=[
            {
                'output': functions_name[action.function.name](json.loads(action.function.arguments)),
                'tool_call_id': action.id
            } for action in required_actions
        ]
    )
    

client = OpenAI(api_key=api_key)


query = st.chat_input(placeholder="Enter the sentence what you want in Wikipedia")

if query:
    with st.status(label="진행중"):
        assistant_id = st.session_state['assistant_id']
        if assistant_id:
            thread = get_thread(role='user', content=query) #thread + message생성
        
            if thread:
                run = get_run(assistant_id=assistant_id, thread_id=thread.id) #run
                run_retrieve = get_run_retrieve(run_id=run.id, thread_id=thread.id)
                while run_retrieve.status == "in_progress" or "queued":
                    time.sleep(1.5)
                    run_retrieve = get_run_retrieve(run_id=run.id, thread_id=thread.id)
                    if run_retrieve.status == "requires_action":
                        submit_tool_outputs(run_id=run.id, thread_id=thread.id, run_retrieve=run_retrieve)
                        run_retrieve = get_run_retrieve(run_id=run.id, thread_id=thread.id)
                        
                    elif run_retrieve.status == "completed":
                        with st.chat_message('ai'):
                            st.write('저장완료')
                        break
                    

                    
                
else:   
    st.markdown(
        '''    
        When you enter a search term, related content is found on Wikipedia.

        The contents are scraped and saved as a text file.
        '''
    )
    st.session_state['assistant_id'] = "asst_QcdEjw3Ocy2yurPKJj1M7Op2"







