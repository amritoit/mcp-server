from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import openai
import numpy as np
from openai import AzureOpenAI
from sklearn.neighbors import NearestNeighbors
import os

my_azure_api_key = "<<YOUR_AZURE_OPENAI_KEY>>"
os.environ["OPENAI_API_KEY"] = my_azure_api_key
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ammondal-llm-test.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"

class RagSample:
    
    def __init__(self, documents, openai_api_key):
        self.documents = documents
        self.embeddings = [self.get_gpt4_embedding_azure(doc) for doc in documents]
        self.embeddings = np.array(self.embeddings)
        # Fit a NearestNeighbors model on the document embeddings using cosine similarity
        self.index = NearestNeighbors(n_neighbors=1, metric='cosine').fit(self.embeddings)        

    #function to generate embeddings using GPT-4.
    def get_gpt4_embedding_azure(self, text):
        # Create a client with your Azure OpenAI endpoint & key
        client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version= os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        # Call embeddings
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small" #deplomyed model name
        )
        # Access the embedding directly from the response object
        return response.data[0].embedding
    
    
    #function to generate embeddings using GPT-4.
    def get_gpt4_embedding(self, text):
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        # Access the embedding directly from the response object
        return response.data[0].embedding
    

    # Function to query the index with a new document/query
    def query_index(self, query):
        query_embedding = self.get_gpt4_embedding_azure(query)
        query_embedding = np.array([query_embedding])
        distance, indices = self.index.kneighbors(query_embedding)
        return self.documents[indices[0][0]]


    def main(self):
        # Get embeddings for each document using the get_gpt4_embedding function
        print("--" * 50)
        print("This is how it looks after going through an embedding model:\n")
        print(self.embeddings)

        # Example Query
        query = "Who will give you the ticket to enjoy the night ?"
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
], openai_api_key=my_azure_api_key).main()
