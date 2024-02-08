import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
import math
import networkx as nx



nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def sentence_similarity(sent1, sent2):
    words1 = [word.lower() for word in sent1 if word.lower() not in stop_words]
    words2 = [word.lower() for word in sent2 if word.lower() not in stop_words]

    all_words = list(set(words1 + words2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in words1:
        vector1[all_words.index(word)] += 1

    for word in words2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)

def cosine_similarity(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    return similarity_matrix


def remove_stopwords(sentence):
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_sentence)



def bm25_similarity(tokenized_sentences):
    k1 = 1.5
    b = 0.75

    sentences = [remove_stopwords(sentence) for sentence in tokenized_sentences]
    corpus = tokenized_sentences
    # Calculate document frequency (DF) for each term
    df = {}
    for doc in sentences:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1

    # Calculate inverse document frequency (IDF) for each term
    num_documents = len(corpus)
    idf = {term: math.log((num_documents - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    # Calculate BM25 scores
    avg_doc_length = sum(len(doc) for doc in corpus) / num_documents
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                doc_length_i = len(corpus[i])
                doc_length_j = len(corpus[j])

                common_terms = set(sentences[i]) & set(sentences[j])

                numerator = sum(idf.get(term, 0) * (corpus[i].count(term) * (k1 + 1)) * (corpus[j].count(term) * (k1 + 1))
                                for term in common_terms)

                denominator_i = sum(sentences[i].count(term) + k1 * (1 - b + b * (doc_length_i / avg_doc_length)) for term in common_terms)
                denominator_j = sum(sentences[j].count(term) + k1 * (1 - b + b * (doc_length_j / avg_doc_length)) for term in common_terms)
                if denominator_i != 0 and denominator_j != 0:
                    similarity_matrix[i][j] = numerator / (denominator_i * denominator_j)
                else:
                    similarity_matrix[i][j] = 0.0  # Handle division by zero


    return similarity_matrix
    

def ensemble_similarity(tokenized_sentences):
    cosine_similarity_matrix = cosine_similarity(tokenized_sentences)
    bm25_similarity_matrix = bm25_similarity(tokenized_sentences)

    # Average the similarity scores
    average_similarity_matrix = (cosine_similarity_matrix + bm25_similarity_matrix) / 2.0

    G = nx.Graph()
    num_sentences = len(average_similarity_matrix)

    for i in range(num_sentences):
        G.add_node(i)  # Add a node for each sentence

    for i in range(num_sentences):
        for j in range(i + 1, num_sentences):
            similarity_score = average_similarity_matrix[i, j]
            G.add_edge(i, j, weight=similarity_score)  # Add an edge between sentences with weight as similarity score
    return G


def generate_summary(graph, sentences, num_sentences):
    ranked_sentences = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    selected_sentences = set()

    for edge in ranked_sentences:
        selected_sentences.add(edge[0])
        if len(selected_sentences) >= num_sentences:
            break
        selected_sentences.add(edge[1])
        if len(selected_sentences) >= num_sentences:
            break

    summary = [sentences[i] for i in selected_sentences]
    return ' '.join(summary)


def summarize(news_text):
    num_sentences = 5
    tokenized_sentences = sent_tokenize(news_text)
    if(len(tokenized_sentences)<=5):
        return news_text
    graph = ensemble_similarity(tokenized_sentences)
    summary = generate_summary(graph, tokenized_sentences, num_sentences)
    return summary
