import streamlit as st
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from textblob import TextBlob
import spacy
import altair as alt
import csv
from wordcloud import WordCloud
from transformers import pipeline
import subprocess
import openai
import matplotlib.pyplot as plt
from streamlit.components.v1 import html

# Set up OpenAI API credentials
openai.api_key = "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"
# Load the English language model for spaCy
nlp = spacy.load('en_core_web_sm')

all_purpose_df = pd.DataFrame()

def generate_wordcloud(text):
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = set(), 
                min_font_size = 10).generate(text)
    return alt.Chart(pd.DataFrame({"text": [text]})).mark_image().encode(
        x=alt.value(400),
        y=alt.value(400),
        image=alt.Image(
            content=wordcloud.to_image(),
            mimetype='image/png'
        )
    ).configure_view(
        strokeWidth=0
    ).properties(
        width=400,
        height=400
    )
# Function to extract reviews from a given link
def extract_reviews(link):
    html_page = urlopen(link)
    soup = BeautifulSoup(html_page)
    reviews = []
    for review in soup.findAll('p', {'class': 'review-text'}):
        reviews.append(review.text)
    return reviews

# Function to analyze sentiment of a given text
def analyze_sentiment(text):
    classifier = pipeline("zero-shot-classification")
    candidate_labels = ["strongly negative", "negative", "neutral", "strongly positive", "positive"]
    #candidate_labels = ["negative", "neutral", "positive"]
    # res = classifier(sequence, candidate_labels)
    # print(res)
    #candidate_labels = ["positive", "negative"]
    hypothesis_template = "The sentiment of this review is {}."
    res = classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
    print(res['labels'][0])
    return res['labels'][0]
def get_sentiment(text):
    # response = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    # messages=[
    #         {"role": "system", "content": "You are a helpful assistant. You are genius in analyzing sentiment of a given text. Please classify the sentiment of the following text into one of the following classes: 'strongly negative', 'weakly negative', 'neutral', 'weakly positive', 'strongly positive'. "},
    #         {"role": "user", "content": "The customer service at this restaurant was terrible. I will never go back."},
    #         {"role": "assistant", "content": "strongly negative"},
    #         {"role": "user", "content": "The food at this restaurant was just okay. Nothing special."},
    #         {"role": "assistant", "content": "weakly negative"},
    #         {"role": "user", "content": "The book I read last night was about space exploration."},
    #         {"role": "assistant", "content": "neutral"},
    #         {"role": "user", "content": "The service at this hotel was good. The staff was friendly."},
    #         {"role": "assistant", "content": "weakly positive"},
    #         {"role": "user", "content": "I'm so impressed with the quality of this product. It exceeded my expectations."},
    #         {"role": "assistant", "content": "strongly positive"},
    #         {"role": "user", "content": text},
    #     ]
    # )
    prompt = f"Please classify the sentiment of the following text into one of the following classes: 'strongly negative', 'negative', 'neutral', 'positive', 'strongly positive'. Answer should be one of these classes nothing else. \n\n{text}\n\nSentiment:"

    # Call the OpenAI API to generate a response to the prompt
    model_engine = "text-davinci-003"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.5,
    )
    # Parse the response and extract the predicted sentiment
    predicted_sentiment = response.choices[0].text.strip() # for GPT3
    #predicted_sentiment = response.choices[0].message['content'].strip()
    return predicted_sentiment
# Function to analyze topics of a given text
def analyze_topics(text, sentiment):
    classifier = pipeline("zero-shot-classification")
    print(sentiment)
    candidate_labels = ["improvement suggestion", "bug", "feature information", "fault or malfunction", "feature request", "information enquiry", "content request"]
    hypothesis_template = "The sentiment of the review is "+sentiment+". The main theme or topic of this review is about {}."
    res = classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
    print(res['labels'][0])
    return res['labels'][0]

# Making compatible csv for plot
def make_csv_compatible(filename):
    sentiment_counts = {}

    # Read CSV file
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Count the number of each sentiment for each topic
        for row in reader:
            sentiment = row['Sentiment']
            topic = row['Topic']
            
            if topic not in sentiment_counts:
                sentiment_counts[topic] = {
                    'strongly positive': 0,
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0,
                    'strongly negative': 0
                }
            
            sentiment_counts[topic][sentiment] += 1

    # Write the CSV file
    with open('topic_sentiment_graph.csv', mode='w', newline='') as csvfile:
        fieldnames = ["Sentiment", "improvement suggestion", "bug", "feature information", "fault", "feature request", "information enquiry", "content request"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Write the sentiment counts for each topic
        for sentiment in ['strongly positive', 'positive', 'neutral', 'negative', 'strongly negative']:
            row = {
                'Sentiment': sentiment.capitalize(),
                'bug': sentiment_counts.get('bug', {}).get(sentiment, 0),
                'fault': sentiment_counts.get('fault or malfunction', {}).get(sentiment, 0),
                'feature information': sentiment_counts.get('feature information', {}).get(sentiment, 0),
                'improvement suggestion': sentiment_counts.get('improvement suggestion', {}).get(sentiment, 0),
                'feature request': sentiment_counts.get('feature request', {}).get(sentiment, 0),
                'information enquiry': sentiment_counts.get('information enquiry', {}).get(sentiment, 0),
                'content request': sentiment_counts.get('content request', {}).get(sentiment, 0),
            }
            writer.writerow(row)
   

# Function to generate summary of a given review
def generate_summary(review):
    summary = ""
    prompt = f"summarize and make a list of important points from the given text \n\n{review}\n\Important points:"

    # Call the OpenAI API to generate a response to the prompt
    model_engine = "text-ada-001"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.5,
    )
    # Parse the response and extract the predicted sentiment
    summary = response.choices[0].text.strip()

    # response = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    # messages=[
    #         {"role": "system", "content": "You are a helpful assistant. You are genius in analyzing sentiment of a given text. summarize and make a list of important points from the given text "},
    #         {"role": "user", "content": "On Samsung devices, there's been a new update that's come through and WhatsApp images do not show the latest image that has been taken, for example like a screenshot that you just took will not be available in the gallery unless you go through and click the three dots and then select the photo. And even when you do that you can always select one photo at a time. Previously all photos would be available even ones that you just took. Please fix this so I can give you a five-star review."},
    #         {"role": "assistant", "content": "1. WhatsApp images on Samsung devices do not show the latest image taken. 2. Screenshots taken will not be available in the gallery. 3. Need to click the three dots and select the photo to view. 4. Only able to select one photo at a time. 5. Previously, all photos were available. 6. Request to fix this issue to receive a five-star review."},
    #         {"role": "user", "content": review},
    #     ]
    # )
    # summary = response.choices[0].message['content'].strip()
    return summary
# Streamlit app
st.set_page_config(page_title="Review Analyzer")
# theme color #006E9E
st.sidebar.title("Steps to analyze reviews")
option = st.sidebar.selectbox("Select an option to extract reviews", ["Extract reviews from a link", "Extract reviews from a CSV file"])

if option == "Extract reviews from a link":
    st.title("App Review Analyzer")

    # Textbox for link input
    link = st.text_input("Enter a link to extract reviews:")

    # Button to extract reviews from link
    if st.button("Extract Reviews"):
        if link:
            reviews = extract_reviews(link)
            sentiment_scores = [analyze_sentiment(review) for review in reviews]
            topic_labels = [analyze_topics(review) for review in reviews]
            df = pd.DataFrame({"Review": reviews, "Sentiment": sentiment_scores, "Topic": topic_labels})
            st.write("Reviews, Sentiments, and Topics:")
            st.write(df)

            # Button to save data to CSV file
            if st.button("Save to CSV"):
                df.to_csv("review_analysis.csv", index=False)
                st.write("Data saved to review_analysis.csv")

        else:
            st.write("Please enter a valid link.")

elif option == "Extract reviews from a CSV file":
    st.title("Preview Reviews")
    df = pd.DataFrame()

    # File upload option to read and display CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.write(df)

        # Button to analyze sentiment of CSV data
    st.sidebar.markdown("### _Step 1: Review sentiment analysis_\n")
    if st.sidebar.button("Analyze Sentiment"):
        # with st.spinner('Please wait...'):
        #     df["Sentiment"] = [analyze_sentiment(row["Review"]) for _, row in df.iterrows()]
        # st.success('Done!')

        # df.to_csv("sentiment_analysis.csv", index=False)
        # all_purpose_df = df
        # st.write("Reviews and Sentiments:")
        # st.write(df[["Review", "Sentiment"]])
        # st.write("Data saved to sentiment_analysis.csv")
        with st.spinner('Please wait...'):
            df = pd.read_csv("sentiment_analysis2.csv")
        st.success('Reviews and Sentiments:')
        st.write(df)
        
        df = pd.read_csv('sentiment_analysis_num.csv', index_col=0)

        # Create a figure and axes with desired size
        fig, ax = plt.subplots(figsize=(15, 7))

        # Define custom colors for each bar
        colors = ['#FF0000', '#006E9E', '#0000FF', '#FFFF00', '#FF00FF']

        # Plot the data with custom colors
        df.plot(kind='bar', ax=ax, color=colors)

        # Customize the plot
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('#Reviews')
        ax.set_title('Distribution of Sentiments for Reviews')
        ax.set_yticks(range(0, df.values.max()+1, 1))

        # Display the plot in Streamlit
        st.pyplot(fig)
        # sentiment_counts = df['Sentiment'].value_counts()
        # chart_data = pd.DataFrame({'Sentiment': sentiment_counts.index, 'Reviews': sentiment_counts.values})
        # chart = alt.Chart(chart_data).mark_bar().encode(
        #     x='Sentiment',
        #     y='Reviews'
        # )
        # st.altair_chart(chart, use_container_width=True)
    
        # Button to analyze topics of CSV data
    st.sidebar.markdown("### _Step 2: Review topic analysis_\n")
    if st.sidebar.button("Analyze Topics"):
        df = pd.read_csv("sentiment_analysis.csv")
        with st.spinner('Please wait...'):
            df["Topic"] = [analyze_topics(row["Review"], row["Sentiment"]) for _, row in df.iterrows()]
        st.success('Reviews and Topics:')
        
        df.to_csv("topic_analysis.csv", index=False)
        st.write("Reviews and Topics:")
        st.write(df[["Review","Sentiment","Topic"]])
        
        # df = df.drop('Review', axis=1)
        # df.to_csv("topic_analysis_dropped.csv", index=False)
        # st.write("Data saved to topic_analysis.csv")
        # make_csv_compatible("topic_analysis_dropped.csv")
        
        
        df = pd.read_csv('topic_sentiment_graph.csv', index_col=0)
        # fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(20, 5)) 
        df.plot(kind='bar', ax=ax)

        # Customize plot
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('#Reviews')
        ax.set_title('Distribution of Sentiments Across Topics')
        ax.set_yticks(range(0, df.values.max()+1, 1))

        # Display plot in Streamlit
        st.pyplot(fig)

        # plt.show()

    st.sidebar.markdown("### _Step 3: Review Summarization_\n")
    if st.sidebar.button("Summarize Reviews"):
        df = pd.read_csv("topic_analysis.csv")
        with st.spinner('Please wait...'):
            df["Summary"] = [generate_summary(row["Review"]) for _, row in df.iterrows()]
        st.success('Reviews and Summary:')
        df.to_csv("review_summary.csv", index=False)
        st.write(df)
        
    #     summary = generate_summary("""
    #     As others have stated, this latest release makes sharing photos from the gallery very difficult, 
    # you can't send multiple photos at once any more, and you can't write captions anymore. 
    # this problem cannot be resolved as it's to do with how whatsapp accesses media from your phone. 
    # Utter garbage, this used to be a great app, now it's practically useless.
        
    #     """)
    #     st.write(summary)
    st.sidebar.markdown("### _Step 4: Mapping reviews to App features_\n")
    if st.sidebar.button("Query to Agent"):
        
    # Button to navigate to another Streamlit app
        subprocess.Popen(["streamlit", "run", "search_by_topic.py"])
        # uploaded_file = st.file_uploader("Choose App Features Documentation", type="csv")
        # if uploaded_file is not None:
        #     documentation_df = pd.read_csv(uploaded_file)

