import numpy as np
import time
import nltk
nltk.download('averaged_perceptron_tagger')

from LDA.utils import create_data, create_data_csr,  count_n_dk, count_n_vk, plot_wordcloud
from LDA.collapsed_gibbs_sampling import collapsed_gibbs_sampling

from collections import Counter

if __name__ == '__main__':
    # X: Bag of word
    # X_words: words
    # y: answer label
    X, words, y, doc_word = create_data_csr()

    # parameters
    n_doc, n_word = int(0.2 * doc_word.shape[0]), doc_word.shape[1]
    n_topic = 4
    alpha, beta = np.ones(n_topic), np.ones(n_word)

    # 20% of all data, flatten
    # X_train = X[:n_doc]
    # X_train = np.ravel(X_train)   # flatten
    print('X_init: (documents, words) ', n_doc, n_word)

    doc_word_train = doc_word[:n_doc]

    # one-hot to word-index
    word_meshgrid, doc_meshgrid = np.meshgrid(
        np.arange(n_word), np.arange(n_doc))

    # mesh grid
    word_meshgrid = np.ravel(word_meshgrid)
    doc_meshgrid = np.ravel(doc_meshgrid)

    # get word index and count
    # word_index = np.repeat(word_meshgrid, X_train)
    # get document index and count
    # doc_index = np.repeat(doc_meshgrid, X_train)
    word_index = np.repeat(doc_word_train.indices, doc_word_train.data)
    doc_index = np.repeat(np.arange(n_doc), np.array(doc_word_train.sum(axis=1))[:, 0])
    # topic of word init by randomize
    topic_init = np.random.randint(
        low=0, high=n_topic, size=sum(doc_word_train.data))

    # count n_dk and n_vk
    # n_dk = count_n_dk(
    #     doc_index, topic_init, n_doc, n_topic)
    # n_vk = count_n_vk(
    #     word_index, topic_init, n_word, n_topic)

    idxes = doc_word_train.sum(axis=1).A1.cumsum()
    prev_idx = 0
    n_dk = np.zeros((n_doc, n_topic))
    for i, idx in enumerate(idxes):
        arr = topic_init[prev_idx:idx]
        for t in range(n_topic):
            n_dk[i, t] = np.count_nonzero(topic_init[prev_idx:idx] == t)
        prev_idx = idx

    n_vk = np.zeros((n_word, n_topic))
    for i, t in enumerate(topic_init):
        idx = word_index[i]
        n_vk[idx, t] += 1

    n_dk_sum = np.sum(n_dk, axis=1) + np.sum(alpha)
    n_vk_sum = np.sum(n_vk, axis=0) + np.sum(beta)

    print('n_dk_init: (document, topic) ', n_dk.shape)
    print('n_vk_init: (words, topic) ', n_vk.shape)

    # Collapsed Gibbs Sampling
    n_iter = 100
    start = time.time()
    topic, n_dk, n_vk = collapsed_gibbs_sampling(
        topic_init, n_iter,
        n_dk, n_vk, doc_index,
        word_index, alpha, beta, n_dk_sum, n_vk_sum)

    end = time.time()-start
    print("end time:", end)

    plot_wordcloud(n_vk, words, n_topic)
