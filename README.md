
Regional Music Recommendation System 🎵


✨ Overview
Regional Music Recommendation System is an interactive app leveraging machine learning to recommend Indian songs tailored to your taste. Through K-Means Clustering and KNN, it suggests similar tracks and learns from your ratings to improve future recommendations.

🚀 Features
🎶 Intelligent Song Recommendations: Input a song, get similar great tracks
📊 Visual Clustering Analysis: Silhouette score plot to reveal cluster effectiveness
⭐ Personalized Feedback: Rate recommended songs (0 to 5 stars)
🔁 Precision Metric: Real-time feedback on system’s accuracy
💾 Persistent Ratings: Your choices & ratings are stored and used to enhance future suggestions

🛠️ Setup
Clone the repository:

BASH

git clone https://github.com/your-username/music-recommender.git
cd music-recommender
Install requirements:

BASH

pip install -r requirements.txt
Add the Dataset:

Place your Indian_Music.csv in the project root.
Dataset must include the columns listed here.
🎵 Usage
Run the app locally:
BASH

streamlit run app.py
Open the Streamlit interface in your browser.
Type any song name in the dataset & get recommendations!
🗃️ Data Requirements
Indian_Music.csv must include at least:

song_name
Popularity
Danceability
Duration(ms)
Energy
Instrumentalness
Key
Liveliness
Loudness(dB)
Mode
Speechiness
Tempo
Time_Signature
Valence(float)
🌟 How it Works
Pipeline:

Feature Normalization
Automatic Optimal Cluster Selection (Silhouette Score)
K-Means Clustering on Songs
KNN for Similarity-based Recommendations
User Feedback Integration + Precision Calculation
📦 File Structure

.
├── app.py
├── Indian_Music.csv
├── user_ratings.csv   # (auto-generated)
├── requirements.txt
└── README.md
🤝 Contributing
PRs are welcome!

Improve the UI/UX 👩‍🎤
Add new features 🎧
Or bring your data! 🥁
📄 License
Distributed under the MIT License. See LICENSE for details.

Enjoy discovering new music! 🎼

