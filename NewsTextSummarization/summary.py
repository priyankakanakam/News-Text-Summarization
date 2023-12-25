import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
import regex as re
nltk.download('punkt')
nltk.download('stopwords')


def read_article(text):
    sentences = sent_tokenize(text)
    return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    words1 = [word.lower() for word in sent1 if word.lower() not in stopwords]
    words2 = [word.lower() for word in sent2 if word.lower() not in stopwords]

    all_words = list(set(words1 + words2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in words1:
        vector1[all_words.index(word)] += 1

    for word in words2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)

    return similarity_matrix


def generate_summary(text, num_sentences):
    stop_words = set(stopwords.words('english'))
    summarize_text = []

    sentences = read_article(text)

    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    sentence_similarity_scores = np.array([sum(row) for row in sentence_similarity_matrix])

    ranked_sentences = [item[0] for item in
                        sorted(enumerate(sentence_similarity_scores), key=lambda x: x[1], reverse=True)]

    for i in range(num_sentences):
        summarize_text.append(sentences[ranked_sentences[i]])

    return ' '.join(summarize_text)

def summarize(news_text):
    # remove emojis
    withoutemoji = news_text.encode('ascii', 'ignore').decode('ascii')
    #print("withoutemoji", withoutemoji)
    # remove extra spaces after period
    removespacesafterperiod = re.sub(r'\.\s+', '. ', withoutemoji)
    #print("removespacesafterperiod", removespacesafterperiod)
    templist = removespacesafterperiod.split(".")
    #print(len(templist), templist)
    # remove null strings in list
    while ("" in templist):
        templist.remove("")
    #print(len(templist), templist)
    cleaned_text = ''
    for i in range(len(templist)):
        cleaned_text += templist[i] + ". "
    #print(cleaned_text)
    summary = ''
    if (len(templist) <= 5):
        summary += generate_summary(cleaned_text,1)
    else:
        num_sentences = round(0.25 * len(templist))
        summary += generate_summary(cleaned_text, num_sentences)
    #print("-------------summary------------")
    #print(summary)
    return summary
