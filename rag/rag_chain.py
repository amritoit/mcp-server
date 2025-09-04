from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import numpy as np
import langchain
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

my_azure_api_key = "<<YOUR_AZURE_API_KEY>>"
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

        # Create a ChromaDB instance
        db = Chroma.from_texts(documents, self.embedding_model)

        # Define the prompt template for generating answers
        self.define_prompt_template()

        # Configure the database to act as a retriever, setting the search type to
        # similarity and returning the top 1 result
        self.retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 1}
        )

        # Create the RAG chain using the retriever, custom prompt, and LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4.1",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} # Pass the context and question
            | self.custom_rag_prompt # Format the prompt using the custom RAG prompt template
            | self.llm # Use the language model to generate a response
            | StrOutputParser() # Parse the output to a string
        )


    def define_prompt_template(self):
        # Define a template for generating answers using provided context
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say 'thanks for asking!' at the end of the answer.

        {context}
        Question: {question}

        Helpful Answer:"""

        # Create a custom prompt template using the defined template, 
        # part of auhgmented generation (RAG) approach
        self.custom_rag_prompt = PromptTemplate.from_template(template)


    def get_azure_openai_query_embedding(self, text):        
        # Call embeddings
        response = self.embedding_model.embed_query(text=text)

        # Access the embedding directly from the response object
        return response


    def main(self):       
        # Example Query
        query ="Who will give you the ticket to enjoy the night ?"
        print("Question:", query)

        # Invoke the RAG chain with a question
        response = self.rag_chain.invoke(query)
        print("Response:", response) # Print the response    


LangChainRag(documents=[
    "This is the Fundamentals of RAG course.",
    "Educative is an AI-powered online learning platform.",
    "There are several Generative AI courses available on Educative.",
    "I am writing this using my keyboard.",
    "JavaScript is a good programming language :)",
    "I will create a ticket in JIRA",
    "I will buy a ticket for the concert"
]).main()