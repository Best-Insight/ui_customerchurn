from sys import set_coroutine_origin_tracking_depth
from time import CLOCK_THREAD_CPUTIME_ID
import streamlit as st
import numpy as np
import pandas as pd
import random
from ui_customerchurn.lda_model import lda_model
from wordcloud import WordCloud
import plotly.express as px

st.markdown("""# NLP: SENTIMENT ANALYSIS
## Identifying Unhappy Customers to Minimize Churn and Increase Retention""")


st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose a csv file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success('File has been successfully uploaded')
    if 'review' not in data.columns:
        st.error('The file need to have a column "review"')

st.markdown("""# NLP ANALYSIS""")


def run_nlp_model(data):
    prediction = [random.randint(0,1) for i in range(data.shape[0])]
    data['recommendation'] = prediction
    st.write(data.head())
    return data

if st.button('Analyse Data'):
    # print is visible in the server output, not in the page
    st.write('I was clicked 🎉')
    data = run_nlp_model(data)
    bad_reviews = data[data['recommendation'] == 0]
    st.write(bad_reviews.head())
else:
    st.write('I was not clicked 😞')


st.markdown("""# LDA ANALYSIS""")

n_topics = st.number_input('Number of Topics', min_value=1, max_value=50, value=5, step=1)
chunks_size = st.number_input('Chunk Size', min_value=1, max_value=1000, value=100, step=50)





if st.button('Run LDA Model'):
    # print is visible in the server output, not in the page
    st.write('I was clicked 🎉')
    lda = lda_model.model(bad_reviews)
    #cloud
    cloud = WordCloud(
        background_color='white',
        max_words=30,
        contour_color='steelblue',
        prefer_horizontal=1.0)

    topics = lda.show_topics(formatted=False)

    for i, topic in enumerate(topics):
        topic_words = dict(topic[1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        st.image(cloud.to_image(), caption=f'Topic #{i}', use_column_width=True)
    #sentences
    #pyldaviz


st.markdown("""# AUTOENCODER REVIEWS VISUALISATION""")

clusters = st.number_input('Select Number of Clusters', min_value=1, max_value=15, value=5, step=1)

category = st.radio('Select a category', ('Realestate', 'Online Courses', 'Bank', 'Insurance'))

def run_auto_encoder_model(bad_reviews, category, n_clusters):
    prediction = [random.randint(0,n_clusters) for i in range(bad_reviews.shape[0])]
    bad_reviews['cluster'] = prediction
    bad_reviews['X'] = bad_reviews['cluster'].apply(lambda row: random.uniform(-1,1))
    bad_reviews['Y'] = bad_reviews['cluster'].apply(lambda row: random.uniform(-1,1))

    px_options = {'hover_data': ['review'], 'color':"cluster"}

    st.write(bad_reviews.head())


    fig = px.scatter(bad_reviews, x='X', y='Y', **px_options)


    st.plotly_chart(fig, use_container_width=True)



if st.button('Run Auto-encoder Visualisation Model'):
    # print is visible in the server output, not in the page
    st.write('I was clicked 🎉')
    data = run_nlp_model(data)
    bad_reviews = data[data['recommendation'] == 0]
    run_auto_encoder_model(bad_reviews, category, clusters)


# butoont(process) NUMBER OF CLUSTERS AND CHUNKS

# return bad or good [0,1,1,1,]

# data['recommended'] = [0,1,1,1,]

# bad_reviews = data[ data[recommended == 0]]


# ENTER FEW PARAMS AND CLICK BUTTON TO SEND A REQUEST TO API FOR LDA

# data_lda = [ word cloud, sentence, pyldavis]

# WORD CLOCK_THREAD_CPUTIME_ID
# seNTENCE
# PYLDAVIS



# AND AUTOENCODER
# SELECT INDUSTRY: REALESTATE, BANK INSURANCE, ONLINE COURSES....
# data_autoencoder = [X, Y]
