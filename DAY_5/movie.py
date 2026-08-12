import streamlit as st
import pickle

cleaned_movies = pickle.load(open("cleaned_movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))
all_movies = pickle.load(open("all_movies.pkl", "rb"))

st.title("Movie Recommender")
selected_movie = st.selectbox(
    "Select a movie",
    all_movies
)
st.write("You selected:", selected_movie)

# ---------------------------------------------------------------------------------

import requests, time

API_KEY = "dd3a1efa85bfd0cb0aaa76a531c03ddb"

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=10)   # add timeout
        response.raise_for_status()                # raise HTTPError if not 200
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except requests.exceptions.RequestException as e:
        print(f" Could not fetch poster for movie_id={movie_id}: {e}")
    return None

def recommend(movie):
    index = cleaned_movies[cleaned_movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movies = []
    recommended_posters = []

    for i in distances[1:6]:
        movie_id = cleaned_movies.iloc[i[0]].movie_id
        recommended_movies.append(cleaned_movies.iloc[i[0]].title)
        poster = fetch_poster(movie_id)
        recommended_posters.append(poster if poster else "No poster available")

    return recommended_movies, recommended_posters


movies, posters = recommend(selected_movie)

# ----------------------------------------------------------------------------------

cols = st.columns(5)

for col, title, poster in zip(cols, movies, posters):
    with col:
        if poster.startswith("http"):
            st.image(poster, width="stretch")
        else:
            st.write("🖼️ No Poster Available")

        st.caption(title)