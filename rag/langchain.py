from langchain_openai import AzureOpenAIEmbeddings
import os
import numpy as np

my_azure_api_key = "<<YOUR_AZURE_OPENAI_KEY>>"
os.environ["OPENAI_API_KEY"] = my_azure_api_key
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ammondal-llm-test.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"

class LangChainRag:
    def __init__(self, documents):
        self.documents = documents
        # Create a client with your Azure OpenAI endpoint & key
        self.client = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version= os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.embeddings = self.client.embed_documents(documents)
        #self.embeddings = np.array(self.embeddings)
        print(len(self.embeddings)) # Print the number of embeddings generated (should be equal to the number of documents)
        print(len((self.embeddings[0]))) # Print the length of the first embedding vector


    def get_azure_openai_query_embedding(self, text):        
        # Call embeddings
        response = self.client.embed_query(text=text)

        # Access the embedding directly from the response object
        return response


    def main(self):       
        # Example Query
        query = "Who will give you the ticket to enjoy the night ?"
        print("Query:", query)
        result = self.get_azure_openai_query_embedding(query)        


LangChainRag(documents=[
    "This is the Fundamentals of RAG course.",
    "Educative is an AI-powered online learning platform.",
    "There are several Generative AI courses available on Educative.",
    "I am writing this using my keyboard.",
    "JavaScript is a good programming language :)",
    "I will create a ticket in JIRA",
    "I will buy a ticket for the concert"
]).main()