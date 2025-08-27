from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import openai


# Importing NearestNeighbors from sklearn for creating the nearest neighbor index
# This module is used to efficiently find the closest vector(s) in high-dimensional space, which is
# crucial for the retrieval functionality in our RAG system
from sklearn.neighbors import NearestNeighbors

class RagSample:
    
    def __init__(self, documents, openai_api_key):
        self.documents = documents
        self.embeddings = [self.get_gpt4_embedding(doc) for doc in documents]
        self.embeddings = np.array(embeddings)
        
        # Fit a NearestNeighbors model on the document embeddings using cosine similarity
        self.index = NearestNeighbors(n_neighbors=1, metric='cosine').fit(embeddings)

        openai.api_key = openai_api_key

    
    #function to generate embeddings using GPT-4.
    def get_gpt4_embedding(text):
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        # Access the embedding directly from the response object
        return response.data[0].embedding
    

    # Function to query the index with a new document/query
    def query_index(query):
        query_embedding = self.get_gpt4_embedding(query)
        query_embedding = np.array([query_embedding])
        distance, indices = self.index.kneighbors(query_embedding)
        return self.documents[indices[0][0]]


    def main(self):
        # Get embeddings for each document using the get_gpt4_embedding function
        print("--" * 50)
        print("This is how it looks after going through an embedding model:\n")
        print(embeddings)

        # Example Query
        query = "What is JS?"
        print("Query:", query)
        result = self.query_index(query) # Retrieve the most similar document to the query
        print("Retrieved document:", result) # Print the retrieved document


RagSample(documents=[
    "This is the Fundamentals of RAG course.",
    "Educative is an AI-powered online learning platform.",
    "There are several Generative AI courses available on Educative.",
    "I am writing this using my keyboard.",
    "JavaScript is a good programming language :)",
    "I will create a ticket in JIRA",
    "I will buy a ticket for the concert"
]).main()
