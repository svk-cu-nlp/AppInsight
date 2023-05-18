import pandas as pd
import streamlit as st
if 'user_input' not in st.session_state:
        st.session_state.user_input = ''
if 'selected_review' not in st.session_state:
        st.session_state.selected_review = ''

def get_text():
    
    st.session_state.user_input = st.session_state.widget
    st.session_state.widget = ''
# Load data
df = pd.read_csv('review_summary.csv')
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
        st.write('Sentiment:', df.loc[selected_row, 'Sentiment'])
        st.write('Topic:', df.loc[selected_row, 'Topic'])
elif topic:
        filtered_df = df[df['Topic'].str.contains(topic, case=False)]
        if not filtered_df.empty:
            
            st.write(filtered_df)
            
            selected_row = st.sidebar.selectbox('Select a row to view details', filtered_df.index, key='review_selector')

            # Display the details of the selected row
            st.write('Review:', filtered_df.loc[selected_row, 'Review'])
            st.session_state.selected_review = filtered_df.loc[selected_row, 'Review']
            st.write('Sentiment:', filtered_df.loc[selected_row, 'Sentiment'])
            st.write('Topic:', filtered_df.loc[selected_row, 'Topic'])

        else:
               st.warning("No data available")
st.markdown("## Upload App documentation and make query\n")
uploaded_file = st.file_uploader("Choose App Features Documentation", type="csv")
if uploaded_file is not None:
    documentation_df = pd.read_csv(uploaded_file)

st.text_input('You:', key='widget', on_change=get_text)
user_input = st.session_state.user_input
st.write('You:', user_input)