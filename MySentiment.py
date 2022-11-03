import streamlit as st
from textblob import TextBlob
import pandas as pd
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

        st.write("""**EDA - Exploratory Data Analysis of Dataset**""")

        if st.button('Dataset Info'):
            info = data.info()
            st.write(info)

        if st.button("Shape of Dataset"):
            shape = data.shape
            st.write(shape)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("No. of Rows", 4025)
        col2.metric("No. of Columns", 5)
        col3.metric("No. of Duplicate values", 4)
        col4.metric("No. of Null Values", 00)
        col5.metric("No. of Missing Values", 00)

        if st.button("Visualization"):
            st.header('Visualization of  number of likes observed till now')
            data1 = data[["Time", "Likes"]]
            plot = data1.plot()
            st.write(plot)

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


if __name__ == '__main__':
    main()