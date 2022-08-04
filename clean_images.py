from PIL import Image
import os
import glob

def resize_image_CNN(path):
    """This function normalises the size and colour mode of every image by finding the minimum height and width of every image and assigning
    the minimum between them to every image where this is just changing the aspect ratio of each image. Additionally, it creates a new directory called cleaned images
    where it stors the high quality images to feed to our CNN Neural Network later on.

    Args:
        path (int): The desired pixel size of new image
        im (image): The image object we want to change size of to normalize
    
    """

    min_width = 100000
    min_height = 100000
    list_img = glob.glob('images/*.jpg')

    new_path = path
    if not os.path.exists(new_path): # if path exists then we will not run the function again
        os.makedirs(new_path)

        for img in list_img:

            size = Image.open(img).size
            width, height = size[0], size[1]

            if width < min_width:
                min_width = width

            if height < min_height:
                min_height = height

        min_height = min(min_height, min_width)
        min_width =  min_height


        # check if cleaned_images exists

        for img in list_img:

            new_img = Image.open(img)
            # Check RGB channels 
            if new_img.mode != 'RGB':
                new_img.mode = 'RGB'
            resized_img = new_img.resize((min_width, min_height))

            path = img.split('\\')[1]
            resized_img.save(f'{new_path}{path}')



def resize_image_ML(final_size, im):
    """This function normalises the size and colour mode of the image to an input final size and RGB mode by pasting every image onto a black background
    of fixed size where it will be used for ML classification.

    Args:
        final_size (int): The desired pixel size of new image
        im (image): The image object we want to change size of to normalize

    Returns:
        image: The new image object formated to required pixel size in RGB mode
    
    """
    size = im.size
    ratio = float(final_size) / max(size) 
    new_image_size = tuple([int(x*ratio) for x in size]) 
    im = im.resize(new_image_size, Image.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size)) # new_img_size = (int(ratio * prev_size[0]), int(ratio * prev_size[1]))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im


def clean_image_data_sklearn(path):
    """This function makes a new directory called cleaned_images_sklearn and saves the resizing of every image with pixel size 50x50 for ML"""
    list_img = glob.glob('images/*.jpg') # Get all files with file type .jpg
    # check if cleaned_images exists
    new_path = path
    if not os.path.exists(new_path):
        os.makedirs(new_path)

        final_size = 50 # We will use 50x50 images when using Sklearn ML libraries

        for item in list_img:
            im = Image.open(item)
            new_im = resize_image_ML(final_size, im)
            file_name = item.split('\\')[1]
            new_im.save(f'{new_path}{file_name}')


if __name__ == '__main__':
    clean_image_data_sklearn('cleaned_images_sklearn/')
    resize_image_CNN('cleaned_images/')
