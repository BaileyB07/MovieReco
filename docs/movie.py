import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os

# Load the movie data
movies = pd.read_csv('archive/tmdb_5000_movies.csv')

# Look at what we have
print(movies.head())
print("\nColumn names:")
print(movies.columns)

# Remove movies without overviews
movies = movies[movies['overview'].notna()]

print(f"\nWe have {len(movies)} movies with descriptions")

# Load the AI model
print("\nLoading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Check if we already have embeddings saved
if os.path.exists('movie_embeddings.pkl'):
    print("Loading saved embeddings...")
    with open('movie_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    print("Loaded!")
else:
    # Create embeddings (this will be slow the first time)
    print("Creating embeddings for all movies...")
    embeddings = model.encode(movies['overview'].tolist(), show_progress_bar=True)
    print(f"\nDone! Created embeddings with shape: {embeddings.shape}")
    
    # Save them for next time
    print("Saving embeddings for next time...")
    with open('movie_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    print("Saved!")

def recommend_movies(movie_titles, top_n=10):
    """
    Give it a list of movie titles you like, get recommendations back
    """
    # Find the movies in our dataset
    user_movies = []
    for title in movie_titles:
        matches = movies[movies['title'].str.lower().str.contains(title.lower(), na=False)]
        if len(matches) > 0:
            user_movies.append(matches.index[0])
            print(f"Found: {matches.iloc[0]['title']}")
        else:
            print(f"Couldn't find: {title}")
    
    if len(user_movies) == 0:
        print("No movies found!")
        return
    
    # Get embeddings for the movies user liked
    user_embeddings = embeddings[user_movies]
    
    # Average them to get user's taste
    user_taste = np.mean(user_embeddings, axis=0).reshape(1, -1)
    
    # Find similar movies
    similarities = cosine_similarity(user_taste, embeddings)[0]
    
    # Get top recommendations (excluding the ones user already listed)
    similar_indices = similarities.argsort()[::-1]
    recommendations = []
    
    for idx in similar_indices:
        if idx not in user_movies and len(recommendations) < top_n:
            recommendations.append(idx)
    
    # Print recommendations
    print(f"\n🎬 Based on your taste, you might like:\n")
    for i, idx in enumerate(recommendations, 1):
        movie = movies.iloc[idx]
        print(f"{i}. {movie['title']} ({movie['release_date'][:4] if pd.notna(movie['release_date']) else 'N/A'})")
        print(f"   {movie['overview']}.\n")

# Test it out! Change these to YOUR favorite movies
print("\n" + "="*50)
recommend_movies(["iron man", "avengers"])