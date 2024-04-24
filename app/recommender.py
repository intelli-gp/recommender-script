from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from dotenv import load_dotenv
from functools import lru_cache

import numpy as np
import pandas as pd

import psycopg2
import os


load_dotenv()


@lru_cache(maxsize=None)
def fetchData():
    #  Connect to database
    try:
        DATABASE_URL = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(DATABASE_URL)
        print("Connected to PostgreSQL database!")
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    cursor = conn.cursor()

    # Fetch Article, article-content, article-tag Table and convert them to a dataframe
    cursor.execute("SELECT * FROM article")
    rows = cursor.fetchall()
    df_articles = pd.DataFrame(rows, columns=[col.name for col in cursor.description])

    cursor.execute("SELECT * FROM article_tag")
    rows = cursor.fetchall()
    df_article_tag = pd.DataFrame(
        rows, columns=[col.name for col in cursor.description]
    )

    cursor.close()
    conn.close()

    return df_articles, df_article_tag


def organize_data_in_df(df_articles, df_article_tag):
    df_article_tag_merged = df_article_tag.groupby("article_id")["tag_name"].agg(list)
    df_article_tag_merged = df_article_tag_merged.to_frame("tags").reset_index()
    df_article_tag_merged["tags"] = [str(x) for x in df_article_tag_merged["tags"]]
    df = pd.merge(
        df_articles,
        df_article_tag_merged[["article_id", "tags"]],
        on="article_id",
        how="left",
    )
    return df


def vectorize_data(df):
    # apply the TfidfVectorizer to the corpus
    corpus = df["tags"]
    corpus = corpus.replace(np.nan, "")
    vectorizer = TfidfVectorizer()
    corpus_vectorized = vectorizer.fit_transform(corpus)

    return corpus_vectorized


def get_article_recommendations(corpus_vectorized, article_id):

    # compute user vector as the average of the vectors of the read articles
    read_articles_rows = []

    # for idx in read_articles_indices:
    article_row = corpus_vectorized.getrow(article_id).toarray()[0]
    read_articles_rows.append(article_row)
    read_articles_rows = np.array(read_articles_rows)
    user_vector_dense = np.average(read_articles_rows, axis=0).reshape((1, -1))
    user_vector = sparse.csr_matrix(user_vector_dense)

    # compute scores as the dot product between the query vector
    # and the documents vectors
    scores = user_vector.dot(corpus_vectorized.transpose())
    scores_array = scores.toarray()[0]
    sorted_indices = scores_array.argsort()[::-1]

    return [[int(idx), round(scores_array[idx], 4)] for idx in sorted_indices]


def start_up():
    df_articles, df_article_tag = fetchData()
    df = organize_data_in_df(df_articles, df_article_tag)
    vectorized_corpus = vectorize_data(df)
    return vectorized_corpus


def main_article(article_id):
    vectorized_corpus = start_up()
    article_scores = get_article_recommendations(vectorized_corpus, article_id)
    return article_scores


# def main_general():
#     vectorized_corpus = start_up()
#     article_scores = get_article_recommendations(vectorized_corpus, 1944)
#     return article_scores
