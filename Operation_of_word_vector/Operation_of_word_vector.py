import numpy as np
from w2v_utils import *

words, word_to_vec_map = read_glove_vecs('glove.6B.60d.txt')

def cosine_similarity(u, v):
    u_norm = np.sqrt(np.sum(np.square(u)))
    v_norm = np.sqrt(np.sum(np.square(v)))
    multi_scale = np.sum(u*v)
    return float(multi_scale)/(u_norm*v_norm)

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    max_simi = -100
    print(word_a, " ", word_b, " ", word_c)
    word_d = None
    for w in words:
        if w in [word_a, word_b, word_c]:
            continue
        # print(word_to_vec_map[word_b]-word_to_vec_map[word_a])

        similar_degree = cosine_similarity(word_to_vec_map[word_b]-word_to_vec_map[word_a], word_to_vec_map[w] - word_to_vec_map[word_c])
        if similar_degree > max_simi:
            max_simi = similar_degree
            word_d = w
    return word_d

def neutralize(word, g, word_to_vec_map):
    e = word_to_vec_map[word]
    e_biascomponent = (np.dot(e,g)/np.linalg.norm(g)**2)*g
    e_debiased = e-e_biascomponent

    return e_debiased

def equalize(pair, bias_axis, word_to_vec_map):
    e1 = word_to_vec_map[pair[0]]
    e2 = word_to_vec_map[pair[1]]
    m = (e1+e2)/2
    mb = (np.dot(m,bias_axis)/np.linalg.norm(bias_axis)**2)*bias_axis
    m_morth = m - mb
    e1b = (np.dot(e1,bias_axis)/np.linalg.norm(bias_axis)**2)*bias_axis
    e2b = (np.dot(e2,bias_axis)/np.linalg.norm(bias_axis)**2)*bias_axis
    e1_equalize = m_morth + np.sqrt(np.abs(1-np.linalg.norm(m_morth)**2))*(e1b-mb)/(np.linalg.norm(e1 - m))
    e2_equalize = m_morth + np.sqrt(np.abs(1-np.linalg.norm(m_morth)**2))*(e2b-mb)/(np.linalg.norm(e2 - m))
    return e1_equalize, e2_equalize


g = word_to_vec_map['woman'] - word_to_vec_map['man']

print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))