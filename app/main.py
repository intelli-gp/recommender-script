from email.header import Header
from fastapi import FastAPI, Depends, Header

from app.recommender import main
from app.auth import get_api_key

app = FastAPI()


@app.get("/")
async def get_article_recommendations(
    api_key: str = Header(None), auth: bool = Depends(get_api_key)
):
    return "hello"


@app.get("/article-recommendations/{id}")
async def get_article_recommendations(
    id: int, api_key: str = Header(None), auth: bool = Depends(get_api_key)
):
    article_scores = main(id, "article", False)
    return article_scores


@app.get("/user-recommendations/{id}")
async def get_article_recommendations(
    id: int, api_key: str = Header(None), auth: bool = Depends(get_api_key)
):
    user_scores = main(id, "user", False)
    return user_scores


@app.get("/general-article-recommendations/{id}")
async def get_article_recommendations(
    id: int, api_key: str = Header(None), auth: bool = Depends(get_api_key)
):
    user_scores = main(id, "article", True)
    return user_scores


@app.get("/general-user-recommendations/{id}")
async def get_article_recommendations(
    id: int, api_key: str = Header(None), auth: bool = Depends(get_api_key)
):
    user_scores = main(id, "user", True)
    return user_scores
