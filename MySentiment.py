import streamlit as st
from textblob import TextBlob
import pandas as pd
import plotly.express as px
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Fxn
def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df


def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)

        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result


def main():
    st.title("Sentiment Analysis NLP App")
    st.subheader("Youtube Comment Analysis")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        data = pd.read_csv("YouTube_Comments.csv")

        if st.button('Show Dataset'):
            st.header('Youtube Comments on iPhone 14 Review')
            st.write(data)

        if st.button("Shape of Dataset"):
            shape = data.shape
            st.write(shape)

        st.subheader("""Dataset Information """)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("No. of Rows", 4025)
        col2.metric("No. of Columns", 5)
        col3.metric("No. of Duplicate values", 4)
        col4.metric("No. of Null Values", 00)
        col5.metric("No. of Missing Values", 00)

        st.subheader("""Visualization""")

        data1 = data[["Time", "Likes"]]

        if st.button("Scatterplot"):
            st.subheader('Visualization between Likes and Time')
            plot1 = px.scatter(data_frame=data1, x="Time", y="Likes")
            st.write(plot1)

        data2 = data.rename({'Reply Count': 'Reply'}, axis=1)
        data2 = data2[["Likes", "Reply"]]

        if st.button("Lineplot"):
            st.subheader('Visualization between Likes and Reply Count')
            plot = px.line(data2, x="Likes", y="Reply")
            st.write(plot)

        st.subheader("Sentiment Analysis of Youtube Comments")

        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # layout
        col1, col2 = st.columns(2)
        if submit_button:

            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                # Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment:: Positive :smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment:: Negative :angry: ")
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c, use_container_width=True)

            with col2:
                st.info("Token Sentiment")

                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)

    else:
        st.subheader("About")
        st.write("""P154 - Youtube Comment Analysis
        
        Group - 1
        
        Mr. Basharnawaj Khan
        
        Mr. Mayuresh Bhoir
        
        Ms. Shruti Dhokale
        
        Mr. Subash S.
        
        Mr. Pranav Bhosale
        
        Mr. Kaustubh Gambare        
         
        Ms. Sarita Aghadate
        
        """)


if __name__ == '__main__':
    main()