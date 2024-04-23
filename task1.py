import os
from flask import Flask, render_template, request, jsonify
import datetime
import speech_recognition as sr
import chardet
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import io
from pydub import AudioSegment
from twilio.rest import Client
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

app = Flask(__name__)

# Azure Blob Storage credentials
connection_string = "DefaultEndpointsProtocol=https;AccountName=placements1;AccountKey=IcnlEGgR4yyEXBaQ49jRYxadeYJlYswJzhSlq8ReqCHjsH464j1e2aPldLGaog038gAePI0J33bZ+ASt8zSdLg==;EndpointSuffix=core.windows.net"
container_name = "threatencalls"

# Initialize Azure Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)
import geocoder

def get_current_location():
    # Get current location using IP address
    g = geocoder.ip('me')
    if g.ok:
        return g.latlng
    else:
        return None

# Get current location
# Load the trained model and vectorizer
def load_model():
    # Detect the encoding of the file
    with open(r"Threaten call detection.csv", 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    
    # Load the data
    df = pd.read_csv(r"Threaten call detection.csv", encoding=encoding)
    df = df.dropna()
    df = df.sample(frac=1, random_state=42)
    
    # Split the data into features and labels
    X = df['Review']
    y = df['Liked']
    
    # Convert text to bag-of-words features
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X)
    
    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y)
    
    return clf, vectorizer

# Initialize the model and vectorizer
clf, vectorizer = load_model()

# Function to predict sentiment and handle audio accordingly
def predict_sentiment(raw_text):
    # Convert raw text to bag-of-words features
    raw_text_bow = vectorizer.transform([raw_text])
    # Use the model to predict the sentiment of the raw text
    sentiment = clf.predict(raw_text_bow)[0]
    return sentiment

@app.route('/')
def index():
    return render_template('index.html')

# Initialize recognizer
r = sr.Recognizer()

@app.route('/record_audio', methods=['GET'])
def record_audio():
    try:
        # Capture audio from microphone
        with sr.Microphone() as source:
            print("Speak something...")
            audio = r.listen(source)
        
        # Recognize speech and convert it to a string
        text = str(r.recognize_google(audio))
        print(text)

        # Save audio file if sentiment is threatening
        sentiment = predict_sentiment(text)
        if sentiment == 0:
            # Save audio to Azure Blob Storage
            audio_data = audio.get_wav_data()
            audio_file_name = f"threatening_call_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"
            blob_client = container_client.get_blob_client(audio_file_name)
            blob_client.upload_blob(audio_data, overwrite=True)
            print("Audio file saved to Azure Blob Storage:", audio_file_name)

        # Send recognized text to process_audio function
        return process_audio(text, sentiment)
    except Exception as e:
        return jsonify({'error': str(e)})

def process_audio(text, sentiment):
    if text is None:
        return jsonify({'error': 'No text data found'})
    
    try:
        # Convert sentiment to a serializable type (e.g., int)
        sentiment = int(sentiment)
        if sentiment == 0:
            account_sid = 'ACfeda23d6bbe33dfc71276a589cf0480b'
            auth_token = '35f727e0ae98c916856d32d4cd37ed59'
            client = Client(account_sid, auth_token)
            current_location = get_current_location()
            if current_location:
                latitude, longitude = current_location
                print("Latitude:", latitude)
                print("Longitude:", longitude)
            else:
                print("Unable to retrieve current location.")
                # Concatenate latitude and longitude to the message body
            body = f'Your Son/Daughter got a threatening call from an unknown number. Location: Latitude {latitude}, Longitude {longitude}.'
            # Create the message with the updated body
            message = client.messages.create(
            from_='+12513085618',
            body=body,
            to='+918978119486')
            print("Message sent:", message.sid)
        return jsonify({'sentiment': sentiment, 'text': text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
