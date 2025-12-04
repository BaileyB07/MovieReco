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

## Setup & Run

### Requirements
```bash
pip install pandas sentence-transformers scikit-learn flask flask-cors
```

### Running the app
1. Start the backend:
```bash
python app.py
```

2. Open `index.html` in your browser

3. Enter movies you like and get recommendations!

## Features
- Smart fuzzy matching (finds movies even with partial names)
- Fast recommendations using cached embeddings
- Clean, modern UI
- Adjustable number of recommendations

## What I learned
- Building ML-powered applications from scratch
- Creating REST APIs with Flask
- Working with embeddings and similarity matching
- Full-stack development basics

---
Made as part of my AI/ML learning journey 🚀
