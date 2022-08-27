import enum
import torchvision.transforms as transforms
import os
import torch
import pandas as pd
from transformers import BertTokenizer
from transformers import BertModel
import torchvision.transforms as transforms
from PIL import Image


class ImageTextDataset(torch.utils.data.Dataset):
    '''
    The ImageTextDataset object inherits its methods from the torch.utils.data.Dataset module.
    It loads all the product description, image ids and category labels from the merged dataframe
    saved from earlier (product_images.csv). Additionally, it reads in the saved decoder file from before
    which helps in decoding the target data into product catgory names. Furthermore, for every index fed to the class
    instance, it gets the corresponding product description and image which is saved in the cleaned_images folder, 
    tranforms the image so it can be fed to the model, tokenizes the product description and creates word embeddings as torch tensor,
    and lastly returns a tuple containing the transformed image, transformed product description, and product catgory label.

    Args:
        root_dir (str): Directory where the merged and cleaned data of products and images is stored
        max_length (int): Maximum length of number of tokens generated in every description
        transform (torchvision.transforms): The transformations applied to a product image before being fed into the model

    '''

    def __init__(self, root_dir: str = 'product_images.csv', max_length: int = 20, transform: transforms = None):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f'The file {self.root_dir} does not exist')

        self.merged_data = pd.read_csv(root_dir)
        self.description = self.merged_data['product_description'].to_list()
        self.labels = self.merged_data['category_codes'].to_list()
        # Get the images   
        self.files = self.merged_data['image_id']

        self.num_classes = len(set(self.labels))
        self.decoder = pd.read_pickle('image_decoder.pkl') # read in the decoder file which we saved from image classification

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.max_length = max_length 

        self.num_classes = len(set(self.labels))

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) 
            ])

    def __getitem__(self, index):
        label = self.labels[index]
        label = torch.as_tensor(label).long()
        image = Image.open('cleaned_images/' + self.files[index] + '.jpg')
        # if image.mode != 'RGB':
        image = self.transform(image).float()
        # else:
        #   image = self.transform(image)
        description = self.description[index]
        encoded = self.tokenizer.batch_encode_plus([description], max_length=self.max_length, padding='max_length', truncation=True) # every description in list format
        encoded = {key: torch.LongTensor(value) for key, value in encoded.items()}

        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)

        description = description.squeeze(0) # remove unnecessary dimension that the CNN model will not use
        return image, description, label

    def __len__(self):
        return len(self.files)


# if __name__ == "__main__":
#     dataset = ImageTextDataset()
#     print(dataset.decoder[int(dataset[2300][2])])
#     print(dataset[2300])
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
#     for i, (image, description, labels) in enumerate(dataloader):
#         # print(image)
#         print(labels)
#         print(image.size())
#         break