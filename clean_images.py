from PIL import Image
import os

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
    """This function makes a new directory called cleaned_images and saves the resizing of every image in that folder
    
    Args:
        path (str): The path to the images folder where we want to resize our product images
    """
    dirs = os.listdir(path)
    # check if cleaned_images exists
    new_path = 'cleaned_images/'
    if not os.path.exists(path+new_path):
        os.makedirs(new_path)

    final_size = 150

    for n, item in enumerate(dirs, 1):
        im = Image.open(path + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'{new_path}{n}_resized.jpg')


if __name__ == '__main__':
    clean_image_data('images/')

