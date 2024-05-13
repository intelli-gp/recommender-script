from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from dotenv import load_dotenv
from functools import lru_cache

import numpy as np
import pandas as pd

import psycopg2
import os


load_dotenv()


class Database:
    def __init__(self):
        DATABASE_HOST = os.getenv("DATABASE_HOST")
        DATABASE_PORT = os.getenv("DATABASE_PORT")
        DATABASE_NAME = os.getenv("DATABASE_NAME")
        DATABASE_USER_NAME = os.getenv("DATABASE_USER_NAME")
        DATABASE_USER_PASSWORD = os.getenv("DATABASE_USER_PASSWORD")

        self.conn = psycopg2.connect(
            database=DATABASE_NAME,
            user=DATABASE_USER_NAME,
            password=DATABASE_USER_PASSWORD,
            host=DATABASE_HOST,
            port=DATABASE_PORT,
        )

    def __enter__(self):
        return self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.conn.close()


# Dependency to inject the database connection into request handlers
def query_database(query):
    with Database() as db:
        db.execute(query)
        rows = db.fetchall()
        return rows, db.description


@lru_cache(maxsize=None)
def fetchDataSpecific(type="article" or "group" or "user"):
    query = f"SELECT * FROM {type}_tag"
    rows, description = query_database(query)
    df_tag = pd.DataFrame(rows, columns=[col.name for col in description])
    return df_tag


@lru_cache(maxsize=None)
def fetchDataGeneral(type="article" or "group" or "user", id=int):
    query = f"SELECT * FROM {type}_tag"
    rows, description = query_database(query)
    df_tag = pd.DataFrame(rows, columns=[col.name for col in description])

    query = f'SELECT * FROM "user_system_tag" where user_id = {id}'
    rows, description = query_database(query)
    user_system_tags = pd.DataFrame(rows, columns=[col.name for col in description])
    user_system_tags[f"{type}_id"] = -1

    query = f'SELECT * FROM "user_tag" where user_id = {id}'
    rows, description = query_database(query)
    user_tags = pd.DataFrame(rows, columns=[col.name for col in description])
    user_tags[f"{type}_id"] = -1

    user_tags = pd.concat([user_tags, user_system_tags], ignore_index=True)

    df_tag = pd.concat(
        [df_tag, user_tags[[f"{type}_id", "tag_name"]]], ignore_index=True
    )
    return df_tag


def organize_data(df_tag, type="article" or "group" or "user"):
    df_tag_merged = df_tag.groupby(f"{type}_id")["tag_name"].agg(list)
    df_tag_merged = df_tag_merged.to_frame("tags").reset_index()
    df_tag_merged["tags"] = [str(x) for x in df_tag_merged["tags"]]
    return df_tag_merged


def vectorize_data(df):
    # apply the TfidfVectorizer to the corpus
    corpus = df["tags"]
    vectorizer = TfidfVectorizer()
    corpus_vectorized = vectorizer.fit_transform(corpus)
    return corpus_vectorized


def get_recommendations(
    corpus_vectorized, data_id, df, type="article" or "group" or "user"
):
    data_rows = []
    data_row = corpus_vectorized.getrow(data_id).toarray()[0]
    data_rows.append(data_row)
    data_rows = np.array(data_rows)
    user_vector_dense = np.average(data_rows, axis=0).reshape((1, -1))
    user_vector = sparse.csr_matrix(user_vector_dense)

    # compute scores as the dot product between the query vector
    # and the documents vectors
    scores = user_vector.dot(corpus_vectorized.transpose())
    scores_array = scores.toarray()[0]
    sorted_indices = scores_array.argsort()[::-1]
    return [
        [int(df.loc[idx, f"{type}_id"]), round(scores_array[idx], 4)]
        for idx in sorted_indices
        if idx != data_id
    ]


def start_up(type="article" or "group" or "user", general=False, id=int):
    if general:
        df_tag = fetchDataGeneral(type, id)
    else:
        df_tag = fetchDataSpecific(type)
    df = organize_data(df_tag, type)
    vectorized_corpus = vectorize_data(df)
    return vectorized_corpus, df


def main(data_id, type="article" or "group" or "user", general=False):
    vectorized_corpus, df = start_up(type, general, data_id)
    if general:
        data_id = 0
    else:
        data_id = df[df[f"{type}_id"] == data_id].index[0]
    scores = get_recommendations(vectorized_corpus, data_id, df, type)
    return scores
