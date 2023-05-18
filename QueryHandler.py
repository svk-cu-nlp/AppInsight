from streamlit_chat import message
import streamlit as st
import pandas as pd
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
def generate_response(user_input):
     content = ""
     if "points" in user_input:
        file = open('example2.txt', 'r')
        content = file.read()
        file.close()

        print(content)
     else:
        file = open('example1.txt', 'r')
        content = file.read()
        file.close()

        print(content)

     return content

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
        
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        user_input = ""
        st.session_state.user_input = ""
else:
    if user_input:
        output = generate_response(user_input)
        st.session_state.q_and_res["query"].append(user_input)
        st.session_state.q_and_res["response"].append(output)
        message(output)
        message(user_input, is_user=True)

        user_input = ""
        st.session_state.user_input = ""