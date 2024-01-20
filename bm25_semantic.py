from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import numpy as np
from numpy.linalg import norm
from bm25 import clean_text, word_segment, normalize_text, remove_stopword, BM25
from extractive_qa_mrc.infer import tokenize_function, data_collator, extract_answer
from extractive_qa_mrc.model.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer
import pandas as pd
import nltk
nltk.download('punkt')

#cosine similarity
def cosine(A, B):
    x=  np.dot(A, B)
    y = norm(A) * norm(B)
    return x/y

docs = []
doc_ids = []
with open('Text/context.txt', encoding='utf-8') as f_r:
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

filename = 'Text/stopwords.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_stopwords = set(data['stopwords'])
dictionary = [[word for word in document.lower().split() if word not in list_stopwords] for document in docs]

model_checkpoint = "nguyenvulebinh/vi-mrc-large"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = MRCQuestionAnswering.from_pretrained(model_checkpoint)

simCSE = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
bm25 = BM25()
bm25.fit(dictionary)

def bm25_search_s(query, limit=3):
    query_processed = clean_text(query)
    query_processed = word_segment(query_processed)
    query_processed = remove_stopword(normalize_text(query_processed))
    query_processed = query_processed.split()

    scores = bm25.search(query_processed)
    scores_index = np.argsort(scores)
    scores_index = scores_index[::-1]

    context = np.array([contents[i] for i in scores_index])[:limit]

    return context

def get_embed(batch_text):
    batch_embedding = simCSE.encode(batch_text)
    return [np.array(vector) for vector in batch_embedding]

def clean_sem(text):
      text = text.lower()
      punc = '''!()[]{};:'\,"”“<>./?@#$%^&*_~'''
      for ele in text:
        if ele in punc:
          text = text.replace(ele, "")
      cleaned_text = ' '.join(text.strip().split())
      return cleaned_text

def reverse_tokenized(tokenized):
      reversed_text = " ".join(tokenized.split("_"))
      reversed_text.replace(" - ", "-")
      reversed_text = reversed_text.replace(" - ", "-")
      return reversed_text

def overlap_splitter(input_string, max_length=256, overlap=20):
    segments = []
    start = 0
    while start < len(input_string):
        end = start + max_length
        segment = input_string[start:end]
        while end < len(input_string) and not input_string[end].isspace():
            end += 1
        segments.append(input_string[start:end].strip())
        start = end - overlap
    return segments

def three_sub_relevant(question, context):
      results = []
      question = clean_sem(question)
      context = clean_sem(context)
      c_tokenized = tokenize(context)
      q_tokenized = tokenize(question)
      chunks = overlap_splitter(c_tokenized)
      q_embed = get_embed(q_tokenized)

      if len(chunks) == 1:
            results.append(context)
            results.append('')
            results.append('')
            return results

      if len(chunks) == 2:
            results.append(reverse_tokenized(chunks[0]))
            results.append(reverse_tokenized(chunks[1]))
            results.append('')
            return results

      embed_chunks = []
      for i in chunks:
            embed_chunks.append(get_embed(i))
      score = [cosine(q_embed, embed_part) for embed_part in embed_chunks]
      top_3_index = np.argsort(score)[::-1][:3]

      results.append(reverse_tokenized(chunks[top_3_index[0]]))
      results.append(reverse_tokenized(chunks[top_3_index[1]]))
      results.append(reverse_tokenized(chunks[top_3_index[2]]))
      return results
