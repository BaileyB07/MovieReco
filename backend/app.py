from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

print("Loading movie data...")
movies = pd.read_csv('archive/tmdb_5000_movies.csv')
movies = movies[movies['overview'].notna()]

print("Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading embeddings...")
with open('movie_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

print("Ready!")

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_titles = data.get('movies', [])
    top_n = data.get('top_n', 10)
    
    if not movie_titles:
        return jsonify({'error': 'No movies provided'}), 400
  
    user_movies = []
    found_movies = []
    
    for title in movie_titles:
        matches = movies[movies['title'].str.lower().str.contains(title.lower(), na=False)]
        if len(matches) > 0:
            user_movies.append(matches.index[0])
            found_movies.append(matches.iloc[0]['title'])
    
    if len(user_movies) == 0:
        return jsonify({'error': 'No matching movies found'}), 404
   
    user_embeddings = embeddings[user_movies]
    user_taste = np.mean(user_embeddings, axis=0).reshape(1, -1)
    similarities = cosine_similarity(user_taste, embeddings)[0]
    
    similar_indices = similarities.argsort()[::-1]
    recommendations = []
    
    for idx in similar_indices:
        if idx not in user_movies and len(recommendations) < top_n:
            movie = movies.iloc[idx]
            recommendations.append({
                'title': movie['title'],
                'year': movie['release_date'][:4] if pd.notna(movie['release_date']) else 'N/A',
                'overview': movie['overview'],
                'rating': float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0
            })
    
    return jsonify({
        'found': found_movies,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)