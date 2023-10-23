import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

import os
import warnings
warnings.filterwarnings('ignore')

data = '../data/potential-talents.xlsx'
df = pd.read_excel(data)

#Load BERT model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

#Encoding:
sen_embeddings = model.encode(df['job_title'])
keyword_embeddings = model.encode(['Aspiring human resources'])
similarity = cosine_similarity(sen_embeddings,keyword_embeddings)
df['fit'] = similarity
df.sort_values(by=['fit'],ascending=False).head()

#rerank with starred candidate: 
starred_candidate_id=17
keyword = str(df.job_title[df.id==starred_candidate_id])
keyword_embeddings = model.encode([keyword])
similarity = cosine_similarity(sen_embeddings,keyword_embeddings)
df['fit'] = similarity
df.sort_values(by=['fit'],ascending=True).head()
