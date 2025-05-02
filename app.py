import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Indian_Music.csv")

    return df

df = load_data()

st.title('AVV Music Recommendation System')

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Popularity', 'Danceability', 'Duration(ms)', 'Energy', 'Instrumentalness', 'Key', 'Liveliness', 'Loudness(dB)', 'Mode', 'Speechiness', 'Tempo', 'Time_Signature', 'Valence(float)']])

# Perform K-Means Clustering
k_range = range(4, 14)
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Select the best number of clusters
best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Song recommendation using KNN
knn = NearestNeighbors(n_neighbors=3)
knn.fit(X_scaled)

# Load or initialize user ratings from CSV file
def load_user_ratings():
    if os.path.exists('user_ratings.csv'):
        return pd.read_csv('user_ratings.csv')
    else:
        return pd.DataFrame(columns=['song_name', 'recommended_song', 'rating'])

user_ratings_df = load_user_ratings()

def recommend_songs(song_name, n_recommendations=5):
    song_index = df[df['song_name'] == song_name].index[0]
    song_vector = X_scaled[song_index].reshape(1, -1)
    distances, indices = knn.kneighbors(song_vector, n_neighbors=n_recommendations + 1)
    
    recommended_songs = df.iloc[indices[0][1:]]['song_name'].values
    
    avg_ratings = np.zeros(len(recommended_songs))
    
    for idx, rec_song in enumerate(recommended_songs):
        song_rating = user_ratings_df[(user_ratings_df['song_name'] == song_name) & 
                                       (user_ratings_df['recommended_song'] == rec_song)]['rating']
        avg_ratings[idx] = song_rating.mean() if not song_rating.empty else 0
    
    combined = list(zip(recommended_songs, avg_ratings))
    combined.sort(key=lambda x: x[1], reverse=True)
    sorted_recommended_songs = [x[0] for x in combined]
    
    return sorted_recommended_songs[:n_recommendations]

# Input song name for recommendation
song_name_input = st.text_input('Enter a song name to get recommendations:', '')
recommended_songs = []

if song_name_input:
    if song_name_input in df['song_name'].values:
        recommendations = recommend_songs(song_name_input)
        recommended_songs.extend(recommendations)
        
        st.write("Please rate the recommendations (0-5):")

        for song in recommended_songs:
            with st.container():
                # Input rating value
                rating_value = st.slider(f"Rate '{song}':", 0, 5, 0, key=song)
                
                # Button to submit rating for each song
                if st.button(f"Submit Rating for '{song}'"):
                    # Save the rating for the song
                    new_rating = pd.DataFrame([[song_name_input, song, rating_value]], columns=['song_name', 'recommended_song', 'rating'])
                    user_ratings_df = pd.concat([user_ratings_df, new_rating], ignore_index=True)
                    user_ratings_df.to_csv('user_ratings.csv', index=False)
                    st.success(f"Rating for '{song}' has been saved!")

        # Calculate precision based on ratings
        true_positive = 0
        false_positive = 0

        for song in recommended_songs:
            song_rating = user_ratings_df[(user_ratings_df['song_name'] == song_name_input) & (user_ratings_df['recommended_song'] == song)]['rating']
            rating = song_rating.mean() if not song_rating.empty else 0
            if rating >= 3:
                true_positive += 1
            elif rating == 1 or rating == 2:
                false_positive += 1

        total_recommendations = len(recommended_songs)
        precision = true_positive / total_recommendations if total_recommendations > 0 else 0

        st.markdown(f"<h3 style='color:white;'>Precision: {precision*100}%</h3>", unsafe_allow_html=True)

    else:
        st.write("**Song not found in the dataset. Please enter a valid song name.**")

# Plot silhouette scores with a partition
st.markdown("---")
st.header("Silhouette Score vs Number of Clusters")
plt.figure(figsize=(8, 5))
plt.plot(list(k_range), silhouette_scores, marker='o')
plt.title("Silhouette Scores for Different Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid()
st.pyplot(plt)
