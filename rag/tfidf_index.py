from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Importing NearestNeighbors from sklearn for creating the nearest neighbor index
# This module is used to efficiently find the closest vector(s) in high-dimensional space, which is
# crucial for the retrieval functionality in our RAG system
from sklearn.neighbors import NearestNeighbors

class RagSample:
    
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Transforming the documents into a matrix of TF-IDF features
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        # Create a DataFrame to display the TF-IDF matrix more readably
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Initializing NearestNeighbors to create a conceptual vector database (index) for the RAG system
        # This index, utilizing cosine similarity, functions effectively as the vector database,
        # storing all document vectors and enabling their retrieval based on similarity measures
        self.index = NearestNeighbors(n_neighbors=1, metric='cosine').fit(self.tfidf_matrix)


    # Function to query the index with a new document/query
    def query_index(self, query):
        # Transforming the query into the same TF-IDF vector space as the documents
        query_vec = self.vectorizer.transform([query])
        print(f"Query Vector for '{query}':")
        print(query_vec.toarray())

        # Finding the nearest neighbor to the query vector in the index
        distance, indices = self.index.kneighbors(query_vec)
        print("Nearest document index:", indices[0][0])
        print("Distance from query:", distance[0][0])

        return self.documents[indices[0][0]]


    def main(self):

        df = pd.DataFrame(self.tfidf_matrix.toarray(), columns=self.feature_names, index=[f"Doc {i+1}" for i in range(len(self.documents))])

        # Print the TF-IDF matrix using DataFrame for better formatting
        print("--" * 50)
        print("This is how it looks after being vectorized:\n")
        print("TF-IDF Matrix:")
        print(df)

        # List of documents to be processed
        # Example query to test the indexing and retrieval system
        query = "Who will own the ticket for the issue we had for similar tickets?" # this fails as we have two ticket document
        result = self.query_index(query)

        # Printing the document retrieved as the closest match to the query
        print("--" * 50)
        print("Retrieved document:", result)

RagSample(documents=[
    "This is the `Fundamentals of RAG course`",
    "Educative is an AI-powered online learning platform",
    "There are several Generative AI courses available on Educative",
    "I am writing this using my keyboard",
    "I will create a ticket in JIRA",
    "I will buy a ticket for the concert"
]).main()
