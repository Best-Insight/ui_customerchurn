from sys import set_coroutine_origin_tracking_depth
from time import CLOCK_THREAD_CPUTIME_ID
import streamlit as st
import numpy as np
import pandas as pd
import random
from ui_customerchurn import lda_model
from wordcloud import WordCloud
from ui_customerchurn import lda_vis
import plotly.express as px
# Sulmaz
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyLDAvis.gensim_models
import pickle
import pyLDAvis
import streamlit.components.v1 as components

# Use the full page instead of a narrow central column # Sulmaz
st.set_page_config(layout="wide")

DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM = 0.001

st.markdown("""# NLP: SENTIMENT ANALYSIS
## Identifying Unhappy Customers to Minimize Churn and Increase Retention""")


st.set_option('deprecation.showfileUploaderEncoding', False)




@st.cache
def get_data():
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success('File has been successfully uploaded')
        if 'review' not in data.columns:
            st.error('The file need to have a column "review"')
        return data


uploaded_file = st.file_uploader("Choose a csv file", type="csv")
data = get_data()
st.markdown("""# NLP ANALYSIS""")


def run_nlp_model(data):
    # prediction = [random.randint(0,1) for i in range(data.shape[0])] Sulmaz
    # data['recommendation'] = prediction Sulmaz
    # st.write(data.head())
    return data


def neg_selector(data): # Sulmaz
    data = run_nlp_model(data)
    bad_reviews = data[data['recommendation'] == 'Not Recommended']
    return bad_reviews

if st.button('Analyse Data'):
    # print is visible in the server output, not in the page
    st.write('I was clicked 🎉')
    data = run_nlp_model(data)
    # bad_reviews = data[data['recommendation'] == 0] Sulmaz
    bad_reviews = data[data['recommendation'] == 'Not Recommended'] # Sulmaz
    st.write(bad_reviews.head())
else:
    st.write('I was not clicked 😞')


#LDA Model
st.markdown("""# LDA ANALYSIS""")

num_topics = st.number_input(label='Number of Topics', min_value=1, max_value=20, value=5, step=1,
    help='The number of requested latent topics to be extracted from the training corpus.')
chunkssize = st.number_input(label= 'Chunk Size',min_value=1, max_value=2000, value=100, step=50,
    help='Number of documents to be used in each training chunk.')

if st.button('Run LDA Model'): # add user input options
    # print is visible in the server output, not in the page
    data = run_nlp_model(data)
    # bad_reviews = data[data['recommendation'] == 'Not Recommended']
    bad_reviews = neg_selector(data)
    lda, corpus = lda_model.model(bad_reviews, num_topics, chunkssize)

    # Word Cloud
    with st.expander('Word Cloud'):
        fig = lda_vis.word_cloud_creator(lda)
        st.pyplot(fig)

    # Topic Highlighted Sentences
    highlight_probability_minimum = st.select_slider(
        'Highlight Probability Minimum',
        options=[10**exponent for exponent in range(-10, 1)],
        value=DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM,
        help=
        'Minimum topic probability in order to color highlight a word in the _Topic Highlighted Sentences_ visualization.'
    )


    with st.expander('Topic Highlighted Sentences'):
        sample = bad_reviews.sample(10)
        lda_vis.highlighted_sentence(sample, lda, highlight_probability_minimum,num_topics)


    #pyldaviz
    with st.expander('Generate pyLDAvis'):
        py_lda_vis_data = pyLDAvis.gensim_models.prepare(lda, corpus, lda.id2word)
        py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)
        components.html(py_lda_vis_html, width=2000, height=800, scrolling=True)





# st.markdown("""# AUTOENCODER REVIEWS VISUALISATION""")

# clusters = st.number_input('Select Number of Clusters', min_value=1, max_value=15, value=5, step=1)

# category = st.radio('Select a category', ('Realestate', 'Online Courses', 'Bank', 'Insurance'))

# def run_auto_encoder_model(bad_reviews, category, n_clusters):
#     prediction = [random.randint(0,n_clusters) for i in range(bad_reviews.shape[0])]
#     bad_reviews['cluster'] = prediction
#     bad_reviews['X'] = bad_reviews['cluster'].apply(lambda row: random.uniform(-1,1))
#     bad_reviews['Y'] = bad_reviews['cluster'].apply(lambda row: random.uniform(-1,1))

#     px_options = {'hover_data': ['review'], 'color':"cluster"}

#     st.write(bad_reviews.head())


#     fig = px.scatter(bad_reviews, x='X', y='Y', **px_options)


#     st.plotly_chart(fig, use_container_width=True)



# if st.button('Run Auto-encoder Visualisation Model'):
#     # print is visible in the server output, not in the page
#     st.write('I was clicked 🎉')
#     data = run_nlp_model(data)
#     bad_reviews = data[data['recommendation'] == 0]
#     run_auto_encoder_model(bad_reviews, category, clusters)


# # butoont(process) NUMBER OF CLUSTERS AND CHUNKS

# # return bad or good [0,1,1,1,]

# # data['recommended'] = [0,1,1,1,]

# # bad_reviews = data[ data[recommended == 0]]


# # ENTER FEW PARAMS AND CLICK BUTTON TO SEND A REQUEST TO API FOR LDA

# # data_lda = [ word cloud, sentence, pyldavis]

# # WORD CLOCK_THREAD_CPUTIME_ID
# # seNTENCE
# # PYLDAVIS



# # AND AUTOENCODER
# # SELECT INDUSTRY: REALESTATE, BANK INSURANCE, ONLINE COURSES....
# # data_autoencoder = [X, Y]
