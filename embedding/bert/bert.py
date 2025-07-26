from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from transformers import BertTokenizer
import torch
from transformers import BertModel
import numpy as np

class Bert:
    """
    A wrapper around the BERT model for text preprocessing, tokenization,
    and embedding generation.
    """

    def __init__(self):
        """
        Initialize the BERT model, tokenizer, stopword list, and lemmatizer.
        """
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()


    def preprocess(self, text):
        """
        Preprocess input text by:
        - Converting to lowercase
        - Tokenizing into words
        - Removing stopwords and punctuation
        - Lemmatizing remaining tokens

        Args:
            text (str): Raw input text.

        Returns:
            str: Preprocessed text string with tokens joined by spaces.
        """
        tokens = word_tokenize(text.lower())
        print("Original text tokens:\n", tokens, "\n")

        tokens = [self.lemmatizer.lemmatize(token)
                  for token in tokens
                  if token not in self.stop_words and token not in string.punctuation]
        print("Tokens after stopword removal and lemmatization:\n", tokens, "\n")

        return ' '.join(tokens)


    def bert_tokenizer(self, preprocessed_text_sequence):
        """
        Tokenize preprocessed text using BERT tokenizer and convert to tensor.

        Args:
            preprocessed_text_sequence (str): Preprocessed input text.

        Returns:
            dict: A dictionary containing tokenized inputs as PyTorch tensors.
        """
        inputs = self.tokenizer(preprocessed_text_sequence,
                                padding=True,
                                truncation=True,
                                return_tensors='pt')
        print("BERT tokenized input:\n", inputs)

        bert_tokenized_text = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print("BERT tokens:\n", bert_tokenized_text)

        return inputs

    
    def generate_embedding(self, inputs):
        """
        Generate BERT embeddings from tokenized input using the last hidden state.

        Args:
            inputs (dict): Tokenized inputs as returned by the tokenizer.

        Returns:
            numpy.ndarray: Word embeddings as a NumPy array.
        """
        bert_tokenized_text = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print("BERT tokens:\n", bert_tokenized_text)
        print("Number of BERT tokens:\n", len(bert_tokenized_text))

        with torch.no_grad():
            outputs = self.model(**inputs)
            word_embeddings = outputs.last_hidden_state.squeeze(0)
            print("Word embeddings shape: ", word_embeddings.shape)
            print("BERT word embeddings: ", word_embeddings)

        return word_embeddings.numpy()


    def run_example(self):
        """
        Main function to demonstrate the use of the Bert class.
        """
        # Initialize the Bert class
        bert_processor = Bert()
        
        Text_sequences = [
            "We had a picnic on the river bank and I went to bank to withdraw money there.",
            "Tom deposited his paycheck into his savings bank account."
        ]       

        preprocessed_text_sequences = [(bert_processor.preprocess(text_sequence)) for text_sequence in Text_sequences]
        print("Preprocessed text: ", preprocessed_text_sequences)

        print("Tokenizing...")
        inputs = bert_processor.bert_tokenizer(preprocessed_text_sequences)

        print("Generating embedding...")
        embedding = bert_processor.generate_embedding(inputs)


# Running example
Bert().run_example()
