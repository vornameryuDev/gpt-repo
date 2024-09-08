import streamlit as st



#---------- headers & icon
st.set_page_config(
    page_title="Document GPT"
)


st.markdown(
    '''
    # Hello!
    ## Welcome to my FullStackGPT Portfolio!

    Here are the apps I made:

    - [DocumentGPT](/DocumentGPT)
    - [QuizGPT](/DocumentGPT)


'''
)

# 검색 버튼과 GitHub 링크 사이의 공간을 많이 띄우기 위해 추가
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