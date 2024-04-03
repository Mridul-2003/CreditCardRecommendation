from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from azure.storage.blob import BlobServiceClient
import os
import io

app = Flask(__name__)

# Load bank data
def load_data():
    # Connect to Azure Blob Storage
    connection_string = "DefaultEndpointsProtocol=https;AccountName=cardrecommenderdata;AccountKey=YRiTpfIbJoWUFeOdmf0TaQDCCz9BDVdkrOXfmnIQM+M5wkVkwBwOd5MFUGW0W3wTWNGzEZDJ/cw8+ASttIr4RQ==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "carddata"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob="CreditBankData.csv")

    # Download CSV data from Azure Blob Storage
    downloaded_bytes = blob_client.download_blob().readall()
    Bank2_data = pd.read_csv(io.BytesIO(downloaded_bytes))
    Bank2_data['Features'].fillna('', inplace=True)
    
    return Bank2_data

Bank2_data = load_data()

# Extract relevant columns for similarity calculation
bank_features = Bank2_data[['Card Name', 'Features']]

# Load TF-IDF vectorizer
def load_tfidf_vectorizer():
    connection_string = "DefaultEndpointsProtocol=https;AccountName=cardrecommenderdata;AccountKey=YRiTpfIbJoWUFeOdmf0TaQDCCz9BDVdkrOXfmnIQM+M5wkVkwBwOd5MFUGW0W3wTWNGzEZDJ/cw8+ASttIr4RQ==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container='cardtfidf', blob="tfidf1_vectorizer.pkl")
    downloaded_bytes = blob_client.download_blob().readall()
    tfidf_vectorizer = pickle.loads(downloaded_bytes)
    return tfidf_vectorizer

tfidf_vectorizer = load_tfidf_vectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(bank_features['Features'])

def recommend_bank(user_inputs):
    # Apply TF-IDF on user input features
    user_feature_text = ' '.join([str(val) for val in user_inputs.values()])
    user_tfidf = tfidf_vectorizer.transform([user_feature_text])

    # Calculate cosine similarity
    user_similarity = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Get top 10 most similar banks
    bank_indices = user_similarity.argsort()[::-1][:10]

    # Get recommended banks
    recommended_banks = bank_features.iloc[bank_indices]
    # Initialize an empty list to hold the image URLs or paths
    image_list = []
    features_list=[]
    # Iterate over the recommended card names
    for card_name in recommended_banks['Card Name'].tolist():
        # Query the Bank2_data DataFrame to find the matching card name and get the Image
        image_url = Bank2_data.loc[Bank2_data['Card Name'] == card_name, 'Image'].values
        features  = Bank2_data.loc[Bank2_data['Card Name'] == card_name,'Features'].values
        features_list.append(features[0])
        if image_url.size > 0:
            # Append the image URL to the list
            image_list.append(image_url[0])
        else:
            image_list.append("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Credit-card-front.svg/512px-Credit-card-front.svg.png?20171209021222")

    return recommended_banks['Card Name'].tolist(), user_similarity[bank_indices], image_list,features_list

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    user_inputs = request.json  # Assuming JSON input
    recommendations, similarity_scores, image_list,features_list = recommend_bank(user_inputs)
    response = {
        "recommended_banks": recommendations,
        "similarity_scores": similarity_scores.tolist(),
        "Images": image_list,
        "Features":features_list,
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
