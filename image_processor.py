from torchvision import transforms
# import os
# from PIL import Image

class ImageProcessor():
    '''
    The class taken in an image, transforms the image in the same fashion as our training and evaluation data so it can be fed to our
    CNN model to output a prediction relating to which product category it belongs to
    '''
    def __init__(self):
        self.transform = transforms.Compose([
        transforms.Resize(128), # change from 155 to 128
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]) 
                    ])

    def __call__(self, image): 
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        image = image[None, :, :, :]
        image = image.reshape(1, 3, 128, 128)
        return image
                

# if __name__ == '__main__':
#     img_dir = os.listdir('cleaned_images')
#     img = img_dir[3]
#     img = Image.open('cleaned_images/'+ img)
#     process = ImageProcessor()
#     img_new = process(img)
#     print(img_new.shape)
