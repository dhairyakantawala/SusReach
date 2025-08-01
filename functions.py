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
from tqdm import tqdm

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

def get_answer(prompt):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    return response.output_text


def process_pdf_get_ans(filepath):
    print("started process")
    df = get_vector_df(filepath)
    template = pd.read_csv('/Users/dhairya/cs projects/SusReach/data/146 Question BRSR Template.csv')
    template['page_number'] = -1

    for i in tqdm(range(len(template))):
        question = template['Reporting Requirement'].iloc[i]
        page_number = page_number_on_embd_df(question, df)
        template.loc[i, 'page_number'] = page_number
    
    print("calculated page numbers")
    
    page_df = get_page_df(filepath)    
    for i in range(len(template)):
        template.loc[i, 'page_content'] = page_df['page_content'].iloc[int(template['page_number'].iloc[i])]
    for i in range(len(template)):
        template.loc[i, 'prompt'] = f"You are a data extraction expert. Given the reporting requirement: '{template['Reporting Requirement'].iloc[i]}' and associated definitions: '{template['Definitions'].iloc[i]}', carefully analyze this BRSR report section: '{template['page_content'].iloc[i]}'. Extract ONLY the specific information requested. Format your response as follows:\n\n1. Use a single line of text\n2. Include only the extracted information without any explanation\n3. Separate multiple items with semicolons\n4. For numerical data, provide exact figures\n5. If information cannot be found, respond only with 'Information not available'\n\nBe precise and concise in your extraction."
    print("started api calling")
    for i in tqdm(range(len(template))):
        try:
            template.loc[i, "answer"] = get_answer(template['prompt'].iloc[i])
        except:
            template.loc[i, "answer"] = "Error caused NA"
    return template[["#Question Ref.", "Reporting Requirement", "Definitions", "Department", "answer"]]