from PIL import Image
import pandas as pd
import os
from merge_data import merge


def resize_image(final_size, im):
    """This function normalises the size and colour mode of the image to an input final size and RGB mode

    Args:
        final_size (int): The desired pixel size of new image
        im (image): The image object we want to change size of to normalize

    Returns:
        image: The new image object formated to required pixel size in RGB mode
    
    """
    size = im.size
    ratio = float(final_size) / max(size) 
    new_image_size = tuple([int(x*ratio) for x in size]) 
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def clean_image_data(path):
    """This function makes a new directory called cleaned_images, sorts the images first, and saves the resizing of every image in that folder
    
    Args:
        path (str): The path to the images folder where we want to resize our product images
    """
    dirs = sorted(os.listdir(path)) # We sort the files here so we have the same order as we would in our tabular dataset for classification later on
    images_data = pd.read_csv('Images.csv').iloc[:, 1:]

    merged_data = merge()

    for i, image_id in enumerate(images_data['id']):
        if image_id not in merged_data['id'].unique():
            images_data.drop(i, inplace=True)

    # check if cleaned_images exists
    new_path = 'cleaned_images/'
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    final_size = 64

    for item in dirs:
        if item.split('.')[0] in list(images_data['id'].unique()):
            im = Image.open(path + item)
            new_im = resize_image(final_size, im)
            new_im.save(f'{new_path}{item}')


if __name__ == '__main__':
    clean_image_data('images/')

