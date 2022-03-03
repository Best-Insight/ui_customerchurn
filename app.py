from re import S
from sys import set_coroutine_origin_tracking_depth
from time import CLOCK_THREAD_CPUTIME_ID
import streamlit as st
import numpy as np
import pandas as pd
import random
from ui_customerchurn import lda_model
from wordcloud import WordCloud
import plotly.express as px
import requests
from unidecode import unidecode
from io import StringIO

from ui_customerchurn import lda_vis


import pyLDAvis.gensim_models
import pyLDAvis
import streamlit.components.v1 as components

from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM = 0.001


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


def run_nlp_model():
    url = 'https://customerchurn1-7fcrmetemq-ew.a.run.app/predict_GCP'
    files = {'file': (uploaded_file.name, StringIO(
    uploaded_file.getvalue().decode("utf-8"))
    , 'multipart/form-data', {'Expires': '0'})}
    response = requests.post(url, files=files)
    prediction = response.json()['pred']
    data['recommendation'] = prediction


tolerance = st.slider('Select tolerance', 0.00, 1.00, 0.50)


@st.cache(allow_output_mutation=True)
def get_bad_reviews():
    run_nlp_model()
    bad_reviews = data[data['recommendation'] < tolerance]
    bad_reviews.to_csv('processed_data.csv')
    return bad_reviews

if st.button('Analyse Data'):
    # print is visible in the server output, not in the page
    bad_reviews = get_bad_reviews()
    st.write(bad_reviews.head())
else:
    st.write('Please click me to process the data')

#LDA Model
st.markdown("""# LDA ANALYSIS""")

num_topics = st.number_input(label='Number of Topics', min_value=1, max_value=20, value=5, step=1,
    help='The number of requested latent topics to be extracted from the training corpus.')
chunkssize = st.number_input(label= 'Chunk Size',min_value=1, max_value=2000, value=100, step=50,
    help='Number of documents to be used in each training chunk.')

if st.button('Run LDA Model'): # add user input options
    # print is visible in the server output, not in the page
    run_nlp_model()
    # bad_reviews = data[data['recommendation'] == 'Not Recommended']
    bad_reviews = get_bad_reviews()
    lda, corpus = lda_model.model(bad_reviews, num_topics, chunkssize)

    # Word Cloud
    with st.expander('Word Cloud'):
        fig = lda_vis.word_cloud_creator(lda)
        st.pyplot(fig)


    with st.expander('Topic Highlighted Sentences'):
        sample = bad_reviews.sample(10)
        lda_vis.highlighted_sentence(sample, lda,
                                     DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM,
                                     num_topics)


    #pyldaviz
    with st.expander('Generate pyLDAvis'):
        py_lda_vis_data = pyLDAvis.gensim_models.prepare(lda, corpus, lda.id2word)
        py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)
        components.html(py_lda_vis_html, width=2000, height=800, scrolling=True)


st.markdown("""# AUTOENCODER REVIEWS VISUALISATION""")

clusters = st.number_input('Select Number of Clusters', min_value=1, max_value=15, value=5, step=1)

category = st.radio('Select a category', ('Financial Services', 'Realestate', 'Online Courses', 'Bank', 'Insurance'))

def run_auto_encoder_model(bad_reviews, category, clusters):
    data_processed = open('processed_data.csv')

    url = 'https://customerchurn1-7fcrmetemq-ew.a.run.app/cluster_finance'
    files = {
        'file': ('data_processed.csv', data_processed, 'multipart/form-data', {
            'Expires': '0'
        })
    }
    response = requests.post(url, files=files)

    prediction_x = response.json()['cluster_x']['X']
    prediction_y = response.json()['cluster_y']['Y']
    bad_reviews['X'] = prediction_x
    bad_reviews['Y'] = prediction_y



    xy_x_train = bad_reviews.reset_index()[['X', 'Y']].values.tolist()
    kmeans = KMeans(n_clusters=clusters,
                    random_state=0).fit(xy_x_train)
    clusters_xy = kmeans.predict(xy_x_train)


    bad_reviews['C'] = clusters_xy

    bad_reviews['review'] = bad_reviews['review'].str.wrap(60)
    bad_reviews['review'] = bad_reviews['review'].apply(
        lambda x: x.replace('\n', '<br>'))

    bad_reviews['review_short'] = bad_reviews['review'].apply(
        lambda x: x if len(x) < 241 else x[:240])



    px_options = {'hover_data': ['review_short'], 'color': "C"}


    fig = px.scatter(bad_reviews, x='X', y='Y', **px_options)

    st.plotly_chart(fig, use_container_width=True)


if st.button('Run Auto-encoder Visualisation Model'):
    # print is visible in the server output, not in the page
    st.write('I was clicked 🎉')

    bad_reviews = get_bad_reviews()
    run_auto_encoder_model(bad_reviews, category, int(clusters))
