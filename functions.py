from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API")
from openai import OpenAI
client = OpenAI(api_key=api_key)
import numpy as np
from numpy.linalg import norm

def get_page_df(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    df = pd.DataFrame([i for i in range(len(pages))])
    for i in range(len(df)):
        df.loc[i, "page_content"] = pages[i].page_content
    df = df.drop(columns=0)
    return df

def get_embd(string_to_convert):
    response = client.embeddings.create(
        input=string_to_convert,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def get_vector_df(filepath):
    df = get_page_df(filepath)
    df['embd'] = df['page_content'].apply(get_embd)
    return df

def cosine_similarity(first, second):
    cosine = np.dot(first, second) / (norm(first) * norm(second))
    return cosine

def page_number_on_embd_df(question_string, embd_df):
    max_index = 0
    max_score = 0
    question_embd = get_embd(question_string)
    for i in range(len(embd_df)):
        score = cosine_similarity(question_embd, embd_df['embd'].iloc[i])
        if score > max_score:
            max_score = score
            max_index = i
    return max_index