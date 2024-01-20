from pyvi.ViTokenizer import tokenize
import re, os, string
import pandas as pd
import math
import numpy as np

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            tf.append(frequencies)

            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query):
        scores = [self._score(query, index) for index in range(self.corpus_size_)]
        return scores

    def _score(self, query, index):
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)

        return score


def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()  # xóa tag của html
    text = re.sub('(\s)+', r'\1', text)  # đưa nhiều dấu cách/tab/xuống dòng thành 1 dấu cách/tab/xuống dòng
    return text


def normalize_text(text):  # Loại bỏ các dấu câu trừ gạch dưới và chuyển thành chữ thường
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()


def word_segment(sent):  # token hóa câu
    sent = tokenize(sent.encode('utf-8').decode('utf-8'))
    return sent


filename = 'Text/stopwords.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_stopwords = data['stopwords']


def remove_stopword(text):  # Loại bỏ stopword
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)
    return text2

context_all_path = 'Text/context.txt'

docs = []
doc_ids = []
with open(context_all_path, encoding='utf-8') as f_r:
    contents = f_r.read().strip().split('======================================================================')
    for content in contents:
        doc_id = content.split(' ')[0].strip()
        if doc_id[0:2] != 'c.':
            doc_id = 'blank'
        content = clean_text(content)
        content = word_segment(content)
        content = normalize_text(content)
        content = remove_stopword(content)
        docs.append(content)
        doc_ids.append(doc_id)

dictionary = [[word for word in document.lower().split() if word not in list_stopwords] for document in docs]


def bm25_search(query, limit=3, k1=1.99, b=0.655):
    bm25 = BM25(k1=k1, b=b)
    bm25.fit(dictionary)
    query_processed = clean_text(query)
    query_processed = word_segment(query_processed)
    query_processed = remove_stopword(normalize_text(query_processed))
    query_processed = query_processed.split()

    scores = bm25.search(query_processed)
    scores_index = np.argsort(scores)
    scores_index = scores_index[::-1]
    scores.sort(reverse=True)
    docs_score = scores[:limit]
    context = np.array([contents[i] for i in scores_index])[:limit]

    return context, docs_score
