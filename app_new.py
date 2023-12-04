import streamlit as st
import sidebar
import textPage_new
# import audioPage
# import imdbReviewsPage
# import imagePage
# import videoPage
# import twitterAnalysisPage

# st.title("Hello")
page = sidebar.show()

if page=="Text Sentiment Analysis":
    textPage_new.renderPage()
# elif page=="Audio":
#     audioPage.renderPage()
# elif page=="IMDb movie reviews":
#     imdbReviewsPage.renderPage()
# elif page=="Image":
#     imagePage.renderPage()
# elif page=="Video":
#     videoPage.main()
# elif page=="Twitter Data":
#     twitterAnalysisPage.renderPage()
