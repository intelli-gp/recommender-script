# Recommender Script

A FastAPI-based recommendation engine that provides tag-based content recommendations for articles, users, and groups using TF-IDF vectorization and cosine similarity.

## Overview

This project implements a scalable recommendation system that analyzes tagged content (articles, users, and groups) and provides personalized recommendations based on tag similarity. It supports both specific and general recommendation modes through a REST API.

## Features

- **Multi-entity Recommendations**: Get recommendations for articles, users, or groups
- **Specific Mode**: Recommendations based on a specific entity's tags
- **General Mode**: Recommendations incorporating user preferences and system tags
- **API Key Authentication**: Secured endpoints with API key validation
- **TF-IDF Based Scoring**: Uses TF-IDF vectorization for intelligent similarity scoring
- **PostgreSQL Integration**: Persistent storage of tag relationships
- **Docker Ready**: Containerized deployment support

## Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- **ML/Data**: scikit-learn, pandas, numpy, scipy
- **Database**: PostgreSQL with psycopg2 driver
- **Environment**: python-dotenv for configuration
- **Deployment**: Docker & Uvicorn ASGI server

## API Endpoints

All endpoints require `api-key` header authentication.

- `GET /` - Health check
- `GET /article-recommendations/{id}` - Article recommendations
- `GET /user-recommendations/{id}` - User recommendations
- `GET /group-recommendations/{id}` - Group recommendations
- `GET /general-article-recommendations/{id}` - Article recommendations with user preferences
- `GET /general-user-recommendations/{id}` - User recommendations with preferences
- `GET /general-group-recommendations/{id}` - Group recommendations with preferences

## Quick Start

1. Clone the repository
2. Create `.env` with database credentials
3. Run: `uvicorn app.main:app --host 0.0.0.0 --port 5000`
