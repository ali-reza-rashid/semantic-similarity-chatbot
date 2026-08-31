# my_script.py
import sys

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify # Library used for running the code on a port
from sentence_transformers import SentenceTransformer # Library used for vectorisation of semantic meaning of questions & input
from sklearn.metrics.pairwise import cosine_similarity # Libary used for comparing the vectors in closeness
import numpy as np

import requests # Used for importing the Q&A from the external airtable source
import json

load_dotenv()

PERSONAL_ACCESS_TOKEN = os.getenv("PERSONAL_ACCESS_TOKEN")
BASE_ID = os.getenv("BASE_ID")
TABLE_NAME = os.getenv("TABLE_NAME")

url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"

headers = {
    "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}"
}

knowledge = {}

try:
    response = requests.get(url, headers=headers) # Gets the data from the table
    
    response.raise_for_status() 
    
    data = response.json() # Parses the JSON response
    
    records = data.get('records', [])
    
    if not records:
        print("No records found in the table. Shutting process down...")
        sys.exit(1)
    else:
        print(f"Found {len(records)} records. Processing...")
        
        for record in records:
            fields = record.get('fields', {})
            
            # Converting the database fields to the dictionary
            question = fields.get('Question')
            answer = fields.get('Answer')
            
            if question:
                knowledge[question] = answer
            else:
                print(f"Skipping record {record.get('id')} - (missing 'Question')")

# Error handling
except requests.exceptions.HTTPError as err:
    print(f"HTTP Error: {err}")
    sys.exit(1)
except requests.exceptions.RequestException as e:
    print(f"A connection error occurred: {e}")
    sys.exit(1)
except KeyError:
    print("Error: Could not parse 'records' from API response.")
    print(f"Response text: {response.text}")
    sys.exit(1)

 
questions = list(knowledge.keys())
answers = list(knowledge.values())
 
app = Flask(__name__)
 
model = SentenceTransformer('all-MiniLM-L6-v2') # Uses the sentence transformers library to understand semantic meaning of a sentence
questionVectors = model.encode(questions) # This vectorises the questions to be able to relate the answers to a semantically similar question asked
 
def question(inputQuestion):
    inputVector = model.encode([inputQuestion]) # This vectorises the input question as done with the preset questions
    semanticSearchResults = cosine_similarity(inputVector, questionVectors) # This does the dot product formula between each of the preset question vectors and the input vector, performed by the scikit-learn library
    closestMatch = np.argmax(semanticSearchResults) # Finds index largest cosine value using the numpy library (since the smallest angle is 0 degrees, which would be the closest in semantic meaning, and cos(0) = 1 whereas the furthest in semantic meaning would be 180 degrees, and cos(180) = -1)
 
    closestMatchValue = semanticSearchResults[0][closestMatch] # Finds the value of the closestMatch

    if (closestMatchValue < 0.4): # This ensures that if the user inputs a question that is not related enough to one of the predefined ones then it does not respond with an irrelevant answer
        return "Sorry, I cannot currently respond to that."
    else:
        return answers[closestMatch]
    
 
@app.route('/', methods=['GET']) # Used as a health check endpoint
def home():
    return "Server running", 200 # Checks if the Flask server is running
 
 
@app.route("/ask", methods=["POST"]) # Receives the JSON payload that contains the user question
def ask():
    data = request.json
    questionText = data.get("question", "")
    answer = question(questionText) # Performs the Semantic Search operation
    return jsonify({"answer": answer}) # Returns answer as JSON
 
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000) # Used to start the Flask server locally and on port 5000