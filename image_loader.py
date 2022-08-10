import torchvision.transforms as transforms
from PIL import Image
import torch
import pandas as pd


class ImageDataset(torch.utils.data.Dataset):
    '''
    The ImageDataset object inherits its methods from the
    torch.utils.data.Dataset module.
    It loads all images from an image folder, creating labels
    for each image according to the subfolder containing the image
    and transform the images so they can be used in a model

    Args:
        transform (torchvision.transforms ): The transformation or list of transformations to be done to the image. If no transform is passed,
        the class will do a generic transformation to resize, convert it to a tensor, and normalize the numbers
    '''

    def __init__(self,
                 transform: transforms = None):

        # # Get the images   
        self.merged_data = pd.read_pickle('product_images.csv')     
        self.files = self.merged_data['image_id']
        self.labels = self.merged_data['category_codes']

        self.num_classes = len(set(self.labels))
        self.decoder = pd.read_pickle('image_decoder.pkl')

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # is this right?
            ])

    def __getitem__(self, index):
        label = self.labels[index]
        label = torch.as_tensor(label).long()
        image = Image.open('cleaned_images/' + self.files[index] + '.jpg')
        # if image.mode != 'RGB':
        image = self.transform(image).float()
        # else:
        #   image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.files)
