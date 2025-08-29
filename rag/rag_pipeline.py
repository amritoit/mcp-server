from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import numpy as np
import langchain


my_azure_api_key = "<<YOUR_AZURE_API_KEY>>"
os.environ["OPENAI_API_KEY"] = my_azure_api_key
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ammondal-llm-test.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"

class LangChainRag:
    def __init__(self, documents):
        self.documents = documents
        # Create a client with your Azure OpenAI endpoint & key
        self.embedding_model = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version= os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.embeddings = self.embedding_model.embed_documents(documents)
        #self.embeddings = np.array(self.embeddings)
        print(len(self.embeddings)) # Print the number of embeddings generated (should be equal to the number of documents)
        print(len((self.embeddings[0]))) # Print the length of the first embedding vector

        # Create a ChromaDB instance
        db = Chroma.from_texts(documents, self.embedding_model)

        # Configure the database to act as a retriever, setting the search type to
        # similarity and returning the top 1 result
        self.retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 1}
        )


    def get_azure_openai_query_embedding(self, text):        
        # Call embeddings
        response = self.embedding_model.embed_query(text=text)

        # Access the embedding directly from the response object
        return response


    def main(self):       
        # Example Query
        query = "Who will give you the ticket to enjoy the night ?"
        print("Query:", query)
        
        # Perform a similarity search with the given query
        result = self.retriever.invoke(query)
        print(result)
        #result = self.get_azure_openai_query_embedding(query)
        #print(result)


LangChainRag(documents=[
    "This is the Fundamentals of RAG course.",
    "Educative is an AI-powered online learning platform.",
    "There are several Generative AI courses available on Educative.",
    "I am writing this using my keyboard.",
    "JavaScript is a good programming language :)",
    "I will create a ticket in JIRA",
    "I will buy a ticket for the concert"
]).main()