import tensorflow_hub as hub
import tensorflow as tf
import numpy as np


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


sentence1 = "people read the book"  # 1.0
sentence2 = "the book people read"  # 0.83875865

# sentence1 = "The bank on the other end of the street was robbed"  # 0.7459192
# sentence2 = "We had a picnic on the bank of the river"  # 0.57716763

# sentence1 = "It is going to be sunny tomorrow"  # 0.7459192
# sentence2 = "It will be fine tomorrow"  # 0.57716763

# sentence1 = "sunny"  # 0.29760534, 0.38802242
# sentence2 = "fine"  #


elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sentence_embeddings = elmo(
    [sentence1, sentence2],
    signature="default",
    as_dict=True)["elmo"]

embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim128/1")
word_embeddings = embed([sentence1, sentence2])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    results_word = sess.run(word_embeddings)
    results_sentence = sess.run(sentence_embeddings)

we1 = results_word[0]
we2 = results_word[1]

se1 = results_sentence[0].sum(axis=0)
se2 = results_sentence[1].sum(axis=0)

print(cos_sim(we1, we2))
print(cos_sim(se1, se2))


