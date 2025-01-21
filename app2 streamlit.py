import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
import numpy as np

page_bg_img = '''
<style>
body {
    background-color: black;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload Excel file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset (replace 'genres_v2.csv' with your file name)
    data = pd.read_csv(uploaded_file)
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*n_init.*")

    # Select the key features for recommendation, including 'genre'
    selected_features = ['song_name', 'danceability', 'energy', 'key', 'loudness', 'mode',
                        'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo', 'genre']

    # Prepare the feature matrix without the song_name column
    X = data[selected_features].drop(columns=['song_name', 'genre'])

    # Fit the Nearest Neighbors model
    model = NearestNeighbors(n_neighbors=5, algorithm='auto')
    model.fit(X)

    # Function to get recommendations based on provided features
    def get_recommendations(features):
        # Convert input features to numeric format
        features = [float(value) for value in features]

        # Find the nearest neighbors based on provided features
        distances, indices = model.kneighbors([features])

        # Display recommended indices, corresponding song names, and genre
        recommended_indices = indices[0]
        recommended_songs = data.iloc[recommended_indices]

        return recommended_songs

    # Streamlit app starts here
    st.title('Song Recommendation System')

    # Sidebar for user input
    st.sidebar.title('Enter Song Features')

    # Create input widgets for each feature
    example_features = []
    for feature in selected_features[1:-1]:  # Excluding 'song_name' and 'genre'
        value = st.sidebar.text_input(f'Enter {feature}', value='0.000000004')
        example_features.append(value)

    if st.sidebar.button('Get Recommendations'):
        recommendations = get_recommendations(example_features)
        st.subheader('Recommended Songs:')
        for i, song in recommendations.iterrows():
            st.write(f"Song Name: {song['song_name']}")
            st.write(f"Genre: {song['genre']}")
            st.write('')

    # Display some information or instructions
    st.write('Enter the values for song features, then click "Get Recommendations" to find similar songs.')
