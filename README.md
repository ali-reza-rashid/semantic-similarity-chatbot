# Semantic Similarity Chatbot

A semantic-search chatbot developed for my A-Level Computer Science NEA.

The application uses sentence embedding and cosine similarity to match a user's question with the most semantically relevant question in an external knowledge base.

Rather than relying on exact keyword matching, the system compares the semantic meaning of questions.

## How It Works

1. Question-and-answer data is retrieved from an Airtable knowledge base using its REST API.

2. The stored questions are converted into vector embeddings using SentenceTransformers and the `all-MiniLM-L6-v2` model.

3. When a user submits a question, the input is converted into an embedding using the same model.

4. Cosine similarity is calculated between the user's question and each stored question.

5. NumPy is used to identify the question with the highest similarity score.

6. If the highest similarity score is below 0.4, the system rejects the match rather than returning a potentially irrelevant answer.

7. If the threshold is met, the answer associated with the closest question is returned to the user.

## Architecture

Airtable Knowledge Base
        ↓
    REST API
        ↓
Question & Answer Data
        ↓
SentenceTransformer
        ↓
Vector Embeddings
        ↓
Cosine Similarity
        ↓
Highest Similarity Score
        ↓
   Threshold ≥ 0.4
      ↙       ↘
   Answer    Rejection

## Technologies Used

- Python
- Flask
- SentenceTransformers
- all-MiniLM-L6-v2
- scikit-learn
- NumPy
- Airtable REST API
- python-dotenv

## Similarity Threshold

An important part of the project was preventing the chatbot from returning an answer simply because one stored question happened to be the closest match.

The system therefore uses a minimum cosine-similarity threshold of 0.4.

If the strongest match falls below this threshold, the chatbot returns:

"Sorry, I cannot currently respond to that."

This reduces hallucinations when the user's question is not sufficiently related to the information contained in the knowledge base.

## API

The application exposes a Flask POST endpoint:

`/ask`

The endpoint accepts a JSON request containing a question and returns
the selected answer as JSON.

Example request:

{
  "question": "Example user question"
}

Example response:

{
  "answer": "Example response"
}

## Security

Airtable credentials are stored as environment variables rather than
being hard-coded into the source code.

The following environment variables are required:

- PERSONAL_ACCESS_TOKEN
- BASE_ID
- TABLE_NAME

The `.env` file is excluded from the repository.

## Project Context

This project was developed as part of my A-Level Computer Science NEA.

**Example knowledge base** (Airtable):

<img width="789" height="513" alt="image" src="https://github.com/user-attachments/assets/dff8e8f4-dde8-4f83-bed6-ec6aa522e9cf" />

Multiple requests must be made if >100 records in Airtable.
