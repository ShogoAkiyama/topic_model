import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import scipy.sparse as ss

import nltk


def create_data():
    vectorizer = CountVectorizer(min_df=0.005, max_df=0.1, stop_words="english")

    categories = ['rec.sport.baseball', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']

    # Get data
    newsgroups_train = fetch_20newsgroups(
        categories=categories, shuffle=True, remove=('headers', 'footers', 'quotes'))
    X = vectorizer.fit_transform(newsgroups_train.data).toarray()
    X_words = np.array(vectorizer.get_feature_names())
    y = newsgroups_train.target

    # select only "NN" or "NNP" or "NNS"
    POS = [nltk.pos_tag(word)[0][1] for word in X_words]
    cond_NN = np.array(list(map(
        lambda x: True if x=='NN' or x=="NNP" or x=="NNS" else False, POS)))
    X_cond = X[:, cond_NN]
    X_words_cond = np.array(X_words)[cond_NN]

    # select the documents which has at least 1 word.
    cond_d_has_1word = np.where(np.sum(X_cond, axis=1) > 0)[0]
    X_cond = X_cond[cond_d_has_1word, :]
    y = y[cond_d_has_1word]

    print('X: (documents, words) ', X_cond.shape)

    return X_cond, X_words_cond, y


def create_data_csr():
    vectorizer = CountVectorizer(min_df=0.005, max_df=0.1, stop_words="english")

    categories = ['rec.sport.baseball', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']

    # Get data
    newsgroups_train = fetch_20newsgroups(
        categories=categories, shuffle=True, remove=('headers', 'footers', 'quotes'))
    X = vectorizer.fit_transform(newsgroups_train.data).toarray()

    words = np.array(vectorizer.get_feature_names())
    y = newsgroups_train.target

    # select only "NN" or "NNP" or "NNS"
    POS = [nltk.pos_tag(word)[0][1] for word in words]
    cond_NN = np.array(list(map(
        lambda x: True if x=='NN' or x=="NNP" or x=="NNS" else False, POS)))
    X_cond = X[:, cond_NN]
    words_cond = np.array(words)[cond_NN]

    # select the documents which has at least 1 word.
    cond_d_has_1word = np.where(np.sum(X_cond, axis=1) > 0)[0]
    X_cond = X_cond[cond_d_has_1word, :]
    y = y[cond_d_has_1word]

    print('X: (documents, words) ', X_cond.shape)

    doc_word = ss.csr_matrix(X_cond)

    return X_cond, words_cond, y, doc_word


def count_n_dk(doc_index, topic_index, n_doc, n_topic):
    """
    Count number of document:d topic:k
    :param doc_index:
    :param topic_index:
    :param n_doc:
    :param n_topic:
    :return:
    """
    n_array = np.zeros((n_doc, n_topic))
    for row, col in zip(doc_index, topic_index):
        n_array[row, col] += 1.0
    return n_array


def count_n_vk(word_index, topic_index, num_word, num_topic):
    n_array = np.zeros((num_word, num_topic))
    for row, col in zip(word_index, topic_index):
        n_array[row, col] += 1.0
    return n_array


def plot_wordcloud(n_kv, news_process1_words, topic_num):
    fig, ax = plt.subplots(1, topic_num, figsize=(12, 6))
    for cate in range(topic_num):
        # log likely hood words
        category_words = news_process1_words[(-n_kv[:, cate]).argsort()][:20]

        # repeat frequency word and add
        word_frequency = np.repeat(
            category_words, (np.sort(n_kv[:, cate])[::-1][:20]*100).astype('int'))
        np.random.shuffle(word_frequency)

        text = ''
        for word in word_frequency:
            text += ' ' + word

        wordcloud = WordCloud(width=480, height=320).generate(text)

        ax[cate].imshow(wordcloud, interpolation='bilinear')
        ax[cate].set_title('topic{}'.format(cate), fontsize=24)
    plt.tight_layout()
    plt.show()
