from wordcloud import WordCloud
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyLDAvis.gensim_models
import pickle
import pyLDAvis
import streamlit.components.v1 as components
import streamlit as st

def word_cloud_creator(model):
    cloud = WordCloud(background_color='white',
                    max_words=50,
                    contour_color='steelblue',
                    prefer_horizontal=1.0,
                    width = 300, height=200)

    topics = model.show_topics(formatted=False)

    fig, axes = plt.subplots(int(len(topics)/3+1),
                                3,
                                sharex=True,
                                sharey=True)
    for i, topic in enumerate(topics):
        topic_words = dict(topic[1])
        cloud.generate_from_frequencies(topic_words, max_font_size=50)
        axes.flatten()[i].imshow(cloud)
        axes.flatten()[i].set_title('Topic ' + str(i), fontdict={'fontsize':15,'fontweight':5})
        axes.flatten()[i].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()

    return fig


def highlighted_sentence(sample, model, highlight_probability_minimum, num_topics):

    COLORS = [color for color in mcolors.XKCD_COLORS.values()]
    colors = random.sample(COLORS,k=num_topics )  #k=model_kwargs['num_topics'] replace it with the number of topics

    for index, row in sample.iterrows():
        html_elements = []
        for token in row['review'].split():
            if model.id2word.token2id.get(token) is None:
                html_elements.append(
                    f'<span style="text-decoration:None;">{token}</span>'
                )
            else:
                term_topics = model.get_term_topics(
                    token, minimum_probability=0)
                topic_probabilities = [
                    term_topic[1] for term_topic in term_topics
                ]
                max_topic_probability = max(
                    topic_probabilities) if topic_probabilities else 0
                if max_topic_probability < highlight_probability_minimum:
                    html_elements.append(token)
                else:
                    max_topic_index = topic_probabilities.index(
                        max_topic_probability)
                    max_topic = term_topics[max_topic_index]
                    background_color = colors[max_topic[0]]
                    color = 'black'
                    html_elements.append(
                        f'<span style="background-color: {background_color}; color: {color}; opacity: 0.5;">{token}</span>'
                    )
        st.markdown(f'Document #{index}: {" ".join(html_elements)}', unsafe_allow_html=True)
