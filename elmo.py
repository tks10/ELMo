import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import argparse

ELMO = "https://tfhub.dev/google/elmo/2"
NNLM = "https://tfhub.dev/google/nnlm-en-dim128/1"


def get_elmo_nnlm_tensors(sentences):
    elmo = hub.Module(ELMO, trainable=True)
    elmo_executable = elmo(
        sentences,
        signature="default",
        as_dict=True)["elmo"]

    nnlm = hub.Module(NNLM)
    nnlm_executable = nnlm(sentences)

    return elmo_executable, nnlm_executable


def get_embeddings(tensors):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        return list(map(sess.run, tensors))


def get_elmo_nnlm_embeddings(sentences):
    return get_embeddings(get_elmo_nnlm_tensors(sentences))


def word_to_sentence(embeddings):
    return embeddings.sum(axis=1)


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def get_parser():
    parser = argparse.ArgumentParser(
        prog='ELMO vs NNLM',
        usage='python elmo.py [sentence1] [sentence2]',
        description='This module demonstrates comparison of ELMo and NNLM',
        add_help=True
    )

    parser.add_argument("sentences", nargs=2, help="The sentences to compare cosine similarity")

    return parser


sentence1 = "people read the book"  # 1.0
sentence2 = "the book people read"  # 0.83875865

# sentence1 = "The bank on the other end of the street was robbed"  # 0.7459192
# sentence2 = "We had a picnic on the bank of the river"  # 0.57716763

# sentence1 = "It is going to be sunny tomorrow"  # 0.7459192
# sentence2 = "It will be fine tomorrow"  # 0.57716763

# Get embeddings corresponding to each sentences
results_elmo, results_nnlm = get_elmo_nnlm_embeddings([sentence1, sentence2])

# Get sentence embeddings by adding. "results_nnlm" is already them.
results_elmo = word_to_sentence(results_elmo)

print("[Cosine Similarity]")
print("\"{}\" vs \"{}\"".format(sentence1, sentence2))
print("ELMO:", cos_sim(results_elmo[0], results_elmo[1]))
print("NNLM:", cos_sim(results_nnlm[0], results_nnlm[1]))
