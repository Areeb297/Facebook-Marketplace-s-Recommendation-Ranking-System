from clean_tabular_data import clean_text_data
import torch
import pandas as pd
from transformers import BertTokenizer
from transformers import BertModel

class TextProcessor():
    '''
    The TextProcessor class takes in a sentence relating to a product, applies the cleaning function we used before to
    clean the description column (remove non-alphanumeric characters etc), tokenizes the text, uses Bert to create word embeddings
    where the max_length is the maximum tokens accepted in a sequence (of every description), and returns encoded text with shape 
    shape (1, 768, 50) 1 is the batch size, 768 is the embedding vector size of each token and 50 is the number of tokens per description.


    Args:
        max_length (int): Maximum length of number of tokens generated in every description

    '''
    def __init__(self, max_length=50):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.max_length = max_length

    def __call__(self, description): 
        '''
        This method runs only when the class instance is called 

        Args:
            description (str): Sample product description text relating to a product category
        '''
        description = clean_text_data(pd.Series(description))[0]
        encoded = self.tokenizer.batch_encode_plus([description], max_length=self.max_length, padding='max_length', truncation=True) # every description in list format
        encoded = {key: torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        description = description.reshape(1, -1, self.max_length)
        return description
                
# if __name__ == '__main__':
#     sample_text = 'This is a new sentence relating to product category bicycle'
#     process = TextProcessor()
#     description = process(sample_text)
#     print(description.shape)