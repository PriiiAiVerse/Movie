import datetime
import os
import json
import pandas as pd
import pickle
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load data
movies = pickle.load(open("movie_dict.pkl", "rb"))
movies = pd.DataFrame(movies)

# Load metadata if available
try:
    metadata = pd.read_csv("movie_metadata.csv")  # movie_id, title, genre, overview, poster_path, cast, vote_average
except:
    metadata = pd.DataFrame()

# Precompute TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['tags'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Data schemas
class LikeData(BaseModel):
    user_id: str
    movie_title: str

class Feedback(BaseModel):
    user_id: str
    movie_id: int
    movie_title: str
    liked: bool = False
    star_rating: int = 0
    watch_duration: float = 0.0
    timestamp: str = datetime.datetime.utcnow().isoformat()

@app.get("/recommend/{movie_title}")
def recommend(movie_title: str):
    movie_title = movie_title.lower().strip()
    index = movies[movies['title'].str.lower().str.strip() == movie_title].index
    if len(index) == 0:
        raise HTTPException(status_code=404, detail="Movie not found")

    index = index[0]
    distances = list(enumerate(cosine_sim[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    results = []
    for i in distances:
        result = metadata[metadata.movie_id == movies.iloc[i[0]].movie_id]
        if not result.empty:
            movie_info = result.iloc[0].to_dict()
            results.append(movie_info)
        else:
            results.append({"title": movies.iloc[i[0]].title})

    return {"recommendations": results}

@app.post("/like")
def like_movie(data: LikeData):
    with open("likes.json", "a") as f:
        f.write(json.dumps(data.dict()) + "\n")
    return {"status": "saved"}

@app.post("/submit_feedback")
def submit_feedback(feedback: Feedback):
    entry = feedback.dict()
    DATA_FILE = "user_feedback.json"

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

    return {"message": "Feedback saved!"}

@app.get("/movie/{movie_id}")
def get_movie_details(movie_id: int):
    result = metadata[metadata.movie_id == movie_id]
    if result.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    return result.iloc[0].to_dict()

@app.get("/search")
def search_movies(q: str):
    result = metadata[metadata['title'].str.contains(q, case=False, na=False)].head(10)
    return result.to_dict(orient="records")

@app.get("/home_sections")
def home_sections():
    if metadata.empty:
        raise HTTPException(status_code=500, detail="Metadata unavailable")

    def section(name, df):
        return {"title": name, "movies": df.head(10).to_dict(orient="records")}

    return {
        "continue_watching": section("Continue Watching", metadata.sample(10)),
        "marvel": section("Marvel Movies", metadata[metadata['title'].str.contains("Marvel", na=False)]),
        "top_10": section("Top 10 Movies", metadata.sort_values(by="vote_average", ascending=False)),
        "your_next_watch": section("Your Next Watch", metadata.sample(10)),
        "genres": {
            genre: metadata[metadata['genre'].str.contains(genre, na=False)].sample(5).to_dict(orient="records")
            for genre in ["Action", "Drama", "Comedy", "Horror", "Sci-Fi"]
        }
    }
