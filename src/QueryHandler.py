from streamlit_chat import message
import streamlit as st
import pandas as pd
import time
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains import VectorDBQA
import os
import argparse
import config
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
OPENAI_API_KEY = config.OPENAI_API_KEY
import openai
openai.organization = config.OPENAI_ORG
openai.api_key = config.OPENAI_API_KEY
# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'q_and_res' not in st.session_state:
    st.session_state.q_and_res = {
        "query": [],
        "response": []
    }
if 'no_doc' not in st.session_state:
    st.session_state['no_doc'] = ""

if 'db' not in st.session_state:
    st.session_state['db'] = ""

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
if 'selected_review' not in st.session_state:
        st.session_state.selected_review = ''


def ask_bot(input_text):
    print("searching DB index")
    result = []
    # persist_directory = st.session_state['db']
    
    persist_directory = './evernotenew_7db3aa5b-dfaf-4dde-8f89-4de0302c5772/db'
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    #print(persist_directory)
    docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    #print(docsearch)
    # vectorstore = Chroma(
    #     collection_name="langchain_store",
    #     embedding_function=embeddings,
    #     persist_directory=persist_directory,
    # )
    #print(docsearch.similarity_search_with_score(query="Give the modified code for getBank method where the database should be used SQL not mongodb.", k=1))
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.schema import(
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    system_template=""" Use the following pieces of context and chat history to answer the users question.
    If you don't know the answer, please just say that you don't know, don't try to make up an answer.
    ---------------
    {context}
    """
    message =[
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(message)
    chain_type_kwargs = {"prompt": prompt}
    search_kwargs = {"k": 1}
    #qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs, search_kwargs=search_kwargs)
    #qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
    #qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
    vectordbkwargs = {"search_distance": 0.9}
    chat_history = []
    try:

        # if st.session_state['no_doc'] > 1:
            # st.write("document greater 1")
        # qa = ChatVectorDBChain.from_llm(ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0), docsearch, qa_prompt=prompt)
        # qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs, search_kwargs=search_kwargs)
        # qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), docsearch.as_retriever(), return_source_documents=False)
        # st.write(qa)
        #result = qa.run(input_text)
        #print(result)
        result = qa({"question": input_text, "chat_history": chat_history})
        # else:
            # st.write("document 1")
        # qa = ChatVectorDBChain.from_llm(ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0), docsearch, qa_prompt=prompt, top_k_docs_for_context=1)
        
        #result = qa.run(input_text)
        #print(result)
        # result = qa({"question": input_text, "chat_history": chat_history})
    except Exception as e:
         st.error(e)
        # qa = ChatVectorDBChain.from_llm(ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0), docsearch, qa_prompt=prompt, top_k_docs_for_context=1)
        # result = qa({"question": input_text, "chat_history": chat_history})

    
    #qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), docsearch.as_retriever(), return_source_documents=False, qa_prompt=prompt)
   
    # return "result"
    return result["answer"]
def generate_response(user_input):
     content = ""
     message = ask_bot(user_input)
   
    #  if "points" in user_input:
    #     file = open('example2.txt', 'r')
    #     content = file.read()
    #     file.close()

    #     print(content)
    #  else:
    #     file = open('example1.txt', 'r')
    #     content = file.read()
    #     file.close()

    #     print(content)

     return message

def get_text():
    st.session_state.user_input = st.session_state.widget
    st.session_state.widget = ''

############################New Code#################################################
# Load data
df = pd.read_csv('review_summary1.csv')
selected_review = ""
# Define dropdown options
options = ["none","improvement suggestion", "bug", "feature addition", "fault", "feature request", "information enquiry", "content request"]

st.sidebar.markdown("### _View Review Analysis by Apply Filters_\n")
# Create sidebar with dropdown
topic = st.sidebar.selectbox('Search by topic', options)
if topic == "none":
        st.write(df)
        selected_row = st.sidebar.selectbox('Select a row to view details', df.index, key='review_selector')

        # Display the details of the selected row
        st.write('Review:', df.loc[selected_row, 'Review'])
        st.session_state.selected_review = df.loc[selected_row, 'Review']
        
elif topic:
        filtered_df = df[df['Topic'].str.contains(topic, case=False)]
        if not filtered_df.empty:
            
            st.write(filtered_df)
            
            selected_row = st.sidebar.selectbox('Select a row to view details', filtered_df.index, key='review_selector')

            # Display the details of the selected row
            st.write('Selected Review:', filtered_df.loc[selected_row, 'Review'])
            st.session_state.selected_review = filtered_df.loc[selected_row, 'Review']
            # st.write('Sentiment:', filtered_df.loc[selected_row, 'Sentiment'])
            # st.write('Topic:', filtered_df.loc[selected_row, 'Topic'])

        else:
               st.warning("No data available")
# st.markdown("## Upload App documentation and make query\n")
# uploaded_file = st.file_uploader("Choose App Features Documentation", type="csv")
# if uploaded_file is not None:
#     documentation_df = pd.read_csv(uploaded_file)



#################################End of new code section #########################

enable_memory = st.checkbox("Enable memory")

st.text_input('Your Query:', key='widget', on_change=get_text)
rev = st.session_state.selected_review
query = "Review: "+rev
user_input = st.session_state.user_input
query = query + " "+user_input

if enable_memory:
    if user_input:
        
        output = generate_response(query)
        # store the output 
        st.session_state.past.append(query)
        st.session_state.generated.append(output)
        st.session_state.q_and_res["query"].append(query)
        st.session_state.q_and_res["response"].append(output)

    if st.session_state['generated']:
        # time.sleep(3)
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        user_input = ""
        st.session_state.user_input = ""
else:
    if user_input:
        output = generate_response(query)
        st.session_state.q_and_res["query"].append(query)
        st.session_state.q_and_res["response"].append(output)
        # time.sleep(3)
        message(output)
        message(user_input, is_user=True)

        user_input = ""
        st.session_state.user_input = ""