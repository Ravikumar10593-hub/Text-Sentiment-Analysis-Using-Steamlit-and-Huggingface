import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob
from PIL import Image
import text2emotion as te
import plotly.graph_objects as go
import emoji
from text2emotion import get_emotion
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
# from flair.models import TextClassifier
# from flair.data import Sentence

# Download NLTK data if not already downloaded
nltk.download('vader_lexicon')

def plotPie(labels, values):
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hoverinfo="label+percent",
            textinfo="value"))
    st.plotly_chart(fig)

def getPolarity(userText):
    tb = TextBlob(userText)
    polarity = round(tb.polarity, 2)
    subjectivity = round(tb.subjectivity, 2)
    if polarity > 0:
        return polarity, subjectivity, "Positive"
    elif polarity == 0:
        return polarity, subjectivity, "Neutral"
    else:
        return polarity, subjectivity, "Negative"

def getSentiments(userText, type):
    if type == 'Positive/Negative/Neutral - TextBlob':
        polarity, subjectivity, status = getPolarity(userText)
        if status == "Positive":
            image = Image.open('./images/positive.PNG')
        elif status == "Negative":
            image = Image.open('./images/negative.PNG')
        else:
            image = Image.open('./images/neutral.PNG')
        col1, col2, col3 = st.columns(3)
        col1.metric("Polarity", polarity, None)
        col2.metric("Subjectivity", subjectivity, None)
        col3.metric("Result", status, None)
        st.image(image, caption=status)
    elif type == 'Happy/Sad/Angry/Fear/Surprise - text2emotion':
        # Remove emojis from userText
        userText_without_emojis = ''.join(c for c in userText if c not in emoji.UNICODE_EMOJI)
        
        # Get emotions from the cleaned text
        emotion = dict(get_emotion(userText_without_emojis))
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Happy 😊", emotion['Happy'], None)
        col2.metric("Sad 😔", emotion['Sad'], None)
        col3.metric("Angry 😠", emotion['Angry'], None)
        col4.metric("Fear 😨", emotion['Fear'], None)
        col5.metric("Surprise 😲", emotion['Surprise'], None)
        plotPie(list(emotion.keys()), list(emotion.values()))

    elif type == 'Positive/Negative/Neutral - VADER':
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(userText)
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            status = "Positive"
            image = Image.open('./images/positive.PNG')
        elif compound_score <= -0.05:
            status = "Negative"
            image = Image.open('./images/negative.PNG')
        else:
            status = "Neutral"
            image = Image.open('./images/neutral.PNG')
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive Score", sentiment_scores['pos'], None)
        col2.metric("Negative Score", sentiment_scores['neg'], None)
        col3.metric("Compound Score", compound_score, None)
#         st.write(f"Result: {status}")
        st.image(image, caption=status)
            
        
    elif type == 'Sentiment Analysis - Hugging Face LLM (distilbert-base-uncased)':
        
        sentiment_label_mapping = {'LABEL_1': 'Positive','LABEL_0': 'Negative'}
        # Example with DistilBERT
        # sentiment_analyzer_distilbert = pipeline("sentiment-analysis", model="assemblyai/distilbert-base-uncased-sst2")
        sentiment_analyzer_distilbert = pipeline("sentiment-analysis", model="distilbert-base-uncased")
        result_distilbert = sentiment_analyzer_distilbert(userText)
        sentiment_label_distilbert = sentiment_label_mapping.get(result_distilbert[0]['label'], 'Neutral')
        sentiment_score_distilbert = result_distilbert[0]['score']
        
        col1, col2 = st.columns(2)
        col1.metric("Sentiment Label", sentiment_label_distilbert, None)
        col2.metric("Sentiment Score", round(sentiment_score_distilbert,2), None)
        if sentiment_label_distilbert == 'Positive':
            st.image(Image.open('./images/positive.PNG'), caption="Positive")
        elif sentiment_label_distilbert == 'Negative':
            st.image(Image.open('./images/negative.PNG'), caption='Negative')
        else:
            st.image(Image.open('./images/neutral.PNG'), caption='Neutral')
        
#     elif type == 'Sentiment Analysis - Hugging Face LLM (bert-base-uncased)':
#         sentiment_label_mapping = {'LABEL_1': 'Positive','LABEL_0': 'Negative'}
#         # Example with DistilBERT
#         # Example with another model (e.g., BERT)
#         sentiment_analyzer_bert = pipeline("sentiment-analysis", model="bert-base-uncased")
#         result_bert = sentiment_analyzer_bert(userText)
#         sentiment_label_bert = sentiment_label_mapping.get(result_bert[0]['label'], 'Neutral')
#         sentiment_score_bert = result_bert[0]['score']
        
#         col1, col2 = st.columns(2)
#         col1.metric("Sentiment Label", sentiment_label_bert, None)
#         col2.metric("Sentiment Score", round(sentiment_score_bert,2), None)
        
        
    elif type == 'Sentiment Analysis - GPT-2':
        sentiment_label_mapping = {'POSITIVE': 'Positive','NEGATIVE': 'Negative'}
        # Example with GPT-2
        sentiment_analyzer_gpt2 = pipeline("sentiment-analysis", model="michelecafagna26/gpt2-medium-finetuned-sst2-sentiment")
        result_gpt2 = sentiment_analyzer_gpt2(userText)
#         sentiment_label_gpt2 = result_gpt2[0]['label']
        sentiment_label_gpt2 = sentiment_label_mapping.get(result_gpt2[0]['label'], 'Neutral')
        sentiment_score_gpt2 = result_gpt2[0]['score']


        col1, col2 = st.columns(2)
        col1.metric("Sentiment Label", sentiment_label_gpt2, None)
        col2.metric("Sentiment Score", round(sentiment_score_gpt2,2), None)
        if sentiment_label_gpt2 == 'Positive':
            st.image(Image.open('./images/positive.PNG'), caption="Positive")
        elif sentiment_label_gpt2 == 'Negative':
            st.image(Image.open('./images/negative.PNG'), caption='Negative')
        else:
            st.image(Image.open('./images/neutral.PNG'), caption='Neutral')

        ## adding Finbert model
    elif type == 'Sentiment Analysis - Finbert':
        sentiment_label_mapping = {'positive': 'Positive','negative': 'Negative', 'neutral' : 'Neutral'}
        # Example with Finbert
        sentiment_analyzer_finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        result_finbert = sentiment_analyzer_finbert(userText)
        # sentiment_label_finbert = result_finbert[0]['label']
        # sentiment_label_finbert = sentiment_label_mapping.get(result_finbert[0]['label'])
        sentiment_score_finbert = result_finbert[0]['score']

        if sentiment_score_finbert >= 0.80:
            sentiment_label_finbert = 'Positive'
        elif sentiment_score_finbert <0.50:
            sentiment_label_finbert = 'Negative'
        else:
            sentiment_label_finbert = 'Neutral'



        col1, col2 = st.columns(2)
        col1.metric("Sentiment Label", sentiment_label_finbert, None)
        col2.metric("Sentiment Score", round(sentiment_score_finbert,2), None)
        if sentiment_label_finbert == 'Positive':
            st.image(Image.open('./images/positive.PNG'), caption="Positive")
        elif sentiment_label_finbert == 'Negative':
            st.image(Image.open('./images/negative.PNG'), caption='Negative')
        else:
            st.image(Image.open('./images/neutral.PNG'), caption='Neutral')
        
        
#     elif type == 'Positive/Negative/Neutral - Flair':
#         classifier = TextClassifier.load('sentiment')
#         sentence = Sentence(userText)
#         classifier.predict(sentence)
#         flair_sentiment = sentence.labels[0].value
#         col1, col2 = st.columns(2)
#         col1.metric("Flair Sentiment", flair_sentiment, None)
#         st.write("Confidence:", sentence.labels[0].score)


# def renderPage():
#     st.title("Sentiment Analysis 😊😐😕😡")
#     components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 10px" /> """)
#     # st.markdown("### User Input Text Analysis")
#     st.subheader("User Input Text Analysis")
#     st.text("Analyzing text data given by the user and find sentiments within it.")
#     st.text("")
#     userText = st.text_input('User Input', placeholder='Input text HERE')
#     st.text("")
#     type = st.selectbox(
#      'Type of analysis',
#      ('Positive/Negative/Neutral - TextBlob', 'Happy/Sad/Angry/Fear/Surprise - text2emotion', 'Positive/Negative/Neutral - VADER', 
#       'Sentiment Analysis - Hugging Face LLM (distilbert-base-uncased)', 'Sentiment Analysis - Hugging Face LLM (bert-base-uncased)', 'Sentiment Analysis - GPT-2'
#      ))
# #     , 'Positive/Negative/Neutral - Flair'))
#     st.text("")
#     if st.button('Predict'):
#         if userText != "" and type != None:
#             st.text("")
#             st.components.v1.html("""
#                                 <h3 style="color: #0284c7; font-family: Source Sans Pro, sans-serif; font-size: 28px; margin-bottom: 10px; margin-top: 50px;">Result</h3>
#                                 """, height=100)
#             getSentiments(userText, type)


# Set page title and background color
st.set_page_config(
    page_title="Sentiment Analysis 📊📈",
    page_icon="😊😐😕😡",
    layout="wide",  # Use wide layout
    initial_sidebar_state="auto",  # Hide the sidebar by default
)

# Add a custom CSS stylesheet for styling
st.markdown(
    """
    <style>
    /* Add your custom CSS styles here */
    body {
        font-family: Arial, sans-serif;
        background-color: #f8f8f8;
        color: #333;
    }
    .stApp {
        max-width: 1200px;  /* Adjust the max width as needed */
        margin: auto;
        padding: 20px;
    }
    h1, h2, h3 {
        color: #0284c7;
        font-family: 'Source Sans Pro', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def renderPage():
    # Main content
    st.title("Sentiment Analysis 😊😐😕😡")

    st.subheader("User Input Text Analysis")
    st.text("Analyzing text data provided by the user to find sentiments within it.")
    st.text("")

    # User input text box
    userText = st.text_area("User Input", value="", height=100)
    st.text("")

    # Selectbox for type of analysis
    type = st.selectbox(
        'Choose the type of analysis',
#         ('Positive/Negative/Neutral - TextBlob', 'Happy/Sad/Angry/Fear/Surprise - text2emotion', 'Positive/Negative/Neutral - VADER',
#          'Sentiment Analysis - Hugging Face LLM (distilbert-base-uncased)', 'Sentiment Analysis - Hugging Face LLM (bert-base-uncased)',
#          'Sentiment Analysis - GPT-2'
#          )
#     )
        
         ('Positive/Negative/Neutral - TextBlob', 'Happy/Sad/Angry/Fear/Surprise - text2emotion', 'Positive/Negative/Neutral - VADER',
         'Sentiment Analysis - Hugging Face LLM (distilbert-base-uncased)',
         'Sentiment Analysis - GPT-2', 'Sentiment Analysis - Finbert'
         )
    )

    st.text("")

    # Predict button
    if st.button('Predict'):
        if userText != "" and type != None:
            st.text("")
            st.components.v1.html("""
                                <h3 style="color: #0284c7; font-family: 'Source Sans Pro', sans-serif; font-size: 28px; margin-bottom: 10px; margin-top: 50px;">Result</h3>
                                """, height=100)
            getSentiments(userText, type)
            
            
            
#             # Add a page break
#             st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
#             st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)

#             # Add a section to compare all models at once
#             st.header("Comparison of all Sentiment Analysis Models")
#             st.text("Here, you can see the results of all sentiment analysis models at once:")
#             st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)

#             st.text(" Result for - Positive/Negative/Neutral - TextBlob:")
#             getSentiments(userText, 'Positive/Negative/Neutral - TextBlob')
#             st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
            
#             st.text(" Result for - Happy/Sad/Angry/Fear/Surprise - text2emotion")
#             getSentiments(userText, 'Happy/Sad/Angry/Fear/Surprise - text2emotion')
#             st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
            
#             st.text(" Result for - Positive/Negative/Neutral - VADER:")
#             getSentiments(userText, 'Positive/Negative/Neutral - VADER')
#             st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
            
#             st.text(" Result for - Sentiment Analysis - Hugging Face LLM (distilbert-base-uncased)")
#             getSentiments(userText, 'Sentiment Analysis - Hugging Face LLM (distilbert-base-uncased)')
#             st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
            
#             st.text(" Result for - Sentiment Analysis - GPT-2:")
#             getSentiments(userText, 'Sentiment Analysis - GPT-2')
#             st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
            
            
            # Add a page break
            st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)

            # Add a section to compare all models at once
            st.header("Comparison of all Sentiment Analysis Models")
            st.text("Here, you can see the results of all sentiment analysis models at once:")
#             st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)

#             st.markdown("**Result for - Positive/Negative/Neutral - TextBlob:**")
            st.markdown('<u><b>Result for - Positive/Negative/Neutral - TextBlob:</b></u>', unsafe_allow_html=True)
            getSentiments(userText, 'Positive/Negative/Neutral - TextBlob')
            st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
            
            st.markdown('<u><b>Result for - Happy/Sad/Angry/Fear/Surprise - text2emotion:</b></u>', unsafe_allow_html=True)
            getSentiments(userText, 'Happy/Sad/Angry/Fear/Surprise - text2emotion')
            st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
            
            st.markdown('<u><b>Result for - Positive/Negative/Neutral - VADER:</b></u>', unsafe_allow_html=True)
            getSentiments(userText, 'Positive/Negative/Neutral - VADER')
            st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
            
            st.markdown('<u><b>Result for - Sentiment Analysis - Hugging Face LLM (distilbert-base-uncased):</b></u>', unsafe_allow_html=True)
            getSentiments(userText, 'Sentiment Analysis - Hugging Face LLM (distilbert-base-uncased)')
            st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
            
            st.markdown('<u><b>Result for - Sentiment Analysis - GPT-2:</b></u>', unsafe_allow_html=True)
            getSentiments(userText, 'Sentiment Analysis - GPT-2')
            st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)

            st.markdown('<u><b>Result for - Sentiment Analysis - Finbert:</b></u>', unsafe_allow_html=True)
            getSentiments(userText, 'Sentiment Analysis - Finbert')
            st.markdown('<hr style="border-top: 3px solid #333;" />', unsafe_allow_html=True)
            
        

