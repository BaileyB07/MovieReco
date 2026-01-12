# 🎬 AI Movie Recommender

An intelligent movie recommendation system that uses AI to suggest movies based on your favorites.

## What it does
Enter movies you like, and the AI will analyze their plots to recommend similar movies you might enjoy.

## Tech Stack
- **Backend**: Python, Flask, Sentence Transformers
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: Sentence embeddings for semantic similarity matching
- **Dataset**: TMDB 5000 Movie Dataset

## How it works
1. Loads 4800+ movies from TMDB dataset
2. Creates AI embeddings of movie plot descriptions
3. When you enter movies you like, it finds similar movies using cosine similarity
4. Returns personalized recommendations

## Why this approach
Instead of a traditional genre or rating based filtering, this project explores embedding-based similarity, allowing recommendations based on deeper semantic meaning in movie plots.

## Features
- Smart fuzzy matching (finds movies even with partial names)
- Fast recommendations using cached embeddings
- Clean, modern UI
- Adjustable number of recommendations

## Limitations
- Recommendations depend heavily on plot quality
- No user personalization or collaborative filtering

## Possible Improvements
- Adding collaborative filtering for personalization
- Introducing evaluation metrics for recommendation quality
- Improving scalability for larger datasets

## What I learned
- Building ML-powered applications from scratch
- Similarity based recommendation techniques
- Working with embeddings and similarity matching
-Integrating ML logic into a simple full-stack application


_This project was built with focus on understanding integration between AI/ML and Full stack rather than optimizing for production._
