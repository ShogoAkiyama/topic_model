import numpy as np


def collapsed_gibbs_sampling(
        z, n_iter, n_dk, n_vk, doc_index,
        word_index, alpha, beta, n_dk_sum, n_vk_sum):

    for s in range(n_iter):
        for idx in range(len(z)):
            doc_id = doc_index[idx]
            word_id = word_index[idx]
            z_id = z[idx]

            # exclude z_id for gibbs_sampling
            n_dk[doc_id, z_id] -= 1.0
            n_vk[word_id, z_id] -= 1.0
            n_vk_sum[z_id] -= 1.0

            # p(z_k| z_\k)
            n_dk_alpha = n_dk[doc_id, :] + alpha
            assert (n_dk_sum[doc_id] - 1 == np.sum(n_dk_alpha)), "value error"
            n_dk_alpha /= (n_dk_sum[doc_id]-1)   # np.sum(n_dk_alpha)

            # p(w_k| z, w_\k)
            n_vk_beta = n_vk[word_id, :] + beta[word_id]
            n_vk_beta /= n_vk_sum   # (np.sum(n_vk, axis=0) + beta[word_id])

            # p(z|w) = p(w_k| z, w_\k) * p(z_k| z_\k)
            sample_p = n_dk_alpha * n_vk_beta
            sample_p /= np.sum(sample_p)

            # sampling from multinomial distribution
            z_new = np.random.multinomial(n=1, pvals=sample_p).argmax()
            z[idx] = z_new

            # update parameters
            n_dk[doc_id, z_new] += 1.0
            n_vk[word_id, z_new] += 1.0
            n_vk_sum[z_new] += 1.0

    return z, n_dk, n_vk
