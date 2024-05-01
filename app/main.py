from email.header import Header
from fastapi import FastAPI, Depends, Header

from app.recommender import main_article
from app.auth import get_api_key

app = FastAPI()


@app.get("/")
async def get_article_recommendations():
    return "hello"


@app.get("/article-recommendations/{id}")
async def get_article_recommendations(
    id: int, api_key: str = Header(None), auth: bool = Depends(get_api_key)
):
    article_scores = main_article(id)
    return article_scores
