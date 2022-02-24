import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import spacy
import en_core_web_sm
import gensim
from unidecode import unidecode
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from gensim.utils import simple_preprocess
import spacy
import nltk
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser
from collections import Counter
from gensim.corpora import Dictionary


def remove_punctuations(text):
    punctuations = string.punctuation
    for punctuation in punctuations:
        text = text.replace(punctuation, '')
    return text

def lowercase(text):
    text = text.lower() #str was removed
    return text

def remove_num(text):
    text = ''.join(word for word in text if not word.isdigit())
    text = unidecode(text)
    return text

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence)))  # deacc=True removes punctuations

def data_words(text):
    data_clean = text['review_clean'].values.tolist() # change to review
    data_words = list(sent_to_words(data_clean))
    return data_words

stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    bigram = Phrases(texts, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


def lemmatization(texts,allowed_postags=['NOUN', 'ADJ']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def common_words_remover (texts):
    flat_list = [item for sublist in texts for item in sublist]
    common_words = [t[0] for t in Counter(flat_list).most_common(20)]
    stop_words.extend(common_words)
    texts = remove_stopwords(texts)
    texts = make_bigrams(texts)
    return texts

def model_params(texts):
    # Create Dictionary
    id2word = Dictionary(texts)
    # Create Corpus
    texts = texts
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    return id2word, corpus

def cleaner(df):

    df['review_clean'] = df['review'].apply(remove_punctuations).copy()
    df['review_clean'] = df['review_clean'].apply(lowercase)
    df['review_clean'] = df['review_clean'].apply(remove_num)
    df['review_clean'] = df['review_clean'].apply(unidecode)
    data_words = data_words(df)
    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops)
    # Do lemmatization keeping only noun, adj
    data_lemmatized = lemmatization(data_words_bigrams,allowed_postags=['NOUN', 'ADJ'])
    data_lemmatized = common_words_remover(data_lemmatized)
    id2word,corpus = model_params(data_lemmatized)
    return id2word, corpus

# Build LDA model
def model(df,num_topics=5,chunksize=100):
    id2word,corpus = cleaner(df)
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)
    return lda_model
