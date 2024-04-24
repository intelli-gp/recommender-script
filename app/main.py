from fastapi import FastAPI

from app.recommender import main_article

app = FastAPI()


@app.get("/")
async def get_article_recommendations():
    return "hello"


@app.get("/article-recommendations/{id}")
async def get_article_recommendations(id: int):
    article_scores = main_article(id)
    return article_scores
