from PIL import Image
import os

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size) 
    new_image_size = tuple([int(x*ratio) for x in size]) 
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def clean_image_data(path):
    dirs = os.listdir(path)
    # check if cleaned_images exists
    if not os.path.exists(path+'cleaned_images'):
        os.makedirs('cleaned_images')
    final_size = 512
    for n, item in enumerate(dirs, 1):
        im = Image.open(path + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'cleaned_images/{n}_resized.jpg')

if __name__ == '__main__':
    clean_image_data('images/')

