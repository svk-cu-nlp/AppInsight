import streamlit as st
import pandas as pd
import csv
import subprocess
import openai
import matplotlib.pyplot as plt
from streamlit.components.v1 import html
import time
import os
import openai
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders.csv_loader import CSVLoader
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
import uuid
import shutil
import config
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
OPENAI_API_KEY = config.OPENAI_API_KEY
import openai
openai.organization = config.OPENAI_ORG
openai.api_key = config.OPENAI_API_KEY

if 'project' not in st.session_state:
    st.session_state['project'] = ""


def create_folder():
    name = st.session_state.widget
    st.session_state.widget = ""
    name = name.replace(" ", "")
    unique_id = str(uuid.uuid4()) # Generate a unique ID for the folder
    folder_name = f"{name}_{unique_id}"
    os.mkdir(folder_name)
    st.session_state.project = folder_name
    return folder_name

def build_knowledge_graph(project_name):
    
    with st.spinner('Please wait...'):
        project_path = os.path.join("./", project_name)
        #project_path = "./source_codes"
        #print(project_path)
        loader = DirectoryLoader(project_path, glob="**/*.txt")
        documents = loader.load()
        # print(documents)
        # st.write("Number of document to be processed: %d" % len(documents))
        # st.session_state['no_doc'] = len(documents)
        text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1500)
        texts = text_splitter.split_documents(documents)
    
        # print("----------------Now Textssss---------------")
        # print(texts)
        persist_directory = os.path.join(project_path,"db")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        st.session_state['db'] = persist_directory
        if os.path.exists(persist_directory):
            # print("----------------Removing all documents---------------")
            shutil.rmtree(persist_directory)
            #print( persist_directory)

        if not os.path.exists(persist_directory):
            docsearch = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
            docsearch.persist()
    st.success('Vector Store for Knowledge Graph has been built successfully')

st.markdown("## Provide Name and Create a Knowledge Graph\n")
st.text_input('Name:', key='widget', on_change=create_folder)

st.markdown("## Upload App documentation\n")
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        try:
            # Decode the file's contents using UTF-8 encoding
            file_contents = uploaded_file.read().decode("ISO-8859-1")

            # Get the filename from the uploaded file object
            actual_filename = uploaded_file.name
            filename = uploaded_file.name+".txt"
            with open(os.path.join(st.session_state.project, filename), "w", encoding="utf-8") as f:
                f.write(file_contents)

            # Display a success message to the user
            st.write(f"File '{actual_filename}' saved successfully!")
        except Exception as e:
            st.warning("Error occurred while processing. Please try again with a different file")
# uploaded_file = st.file_uploader("Choose App Features Documentation", type="csv")
# # print(uploaded_file)

# if uploaded_file is not None:
#     # documentation_df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    
if st.button("Build Knowledge Graph"):
    project = st.session_state.project
    
    build_knowledge_graph(project)
        
if st.button("Start Conversation"):
    project = st.session_state.project
    project = 'evernotenew_7db3aa5b-dfaf-4dde-8f89-4de0302c5772'
    subprocess.Popen(["streamlit", "run", "QueryHandler.py", "--input", project])
