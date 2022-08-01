import pandas as pd
import numpy as np
from clean_tabular_data import get_data_pipeline, clean_time
import os
from PIL import Image
import pickle

def merge():
    """This function merges the product and image datasets where every product can have more than one image

    Returns:
        pd.DataFrame: We return a cleaned dataframe ready to use for image classification
    """

    products_data = get_data_pipeline()
    images_data = pd.read_csv("images.csv")
    images_data.drop(columns='Unnamed: 0', inplace=True)

    images_data['create_time'] = clean_time(images_data['create_time'])

    df = pd.merge(left=products_data, right=images_data, on=('product_id', 'create_time'), sort=True)
    df.drop(columns=['product_id','create_time', 'bucket_link', 'image_ref'], inplace=True)
    df.rename(columns={'id': 'image_id'}, inplace=True)

    return df

def merge_im_array():
    """This function gets the images from the cleaned_images folder, converts them to an array and merges them into the 
    main (merged) dataframe whilst also checking that the each image array corresponds to the correct image id
    """

    data = merge()
    dirs = os.listdir('cleaned_images/')

    data['image_array'] = 'None' # initalize empty column in dataframe

    for item in dirs:
        
        image = Image.open('cleaned_images/' + item)
        arr_im = np.asarray(image)
        data['image_array'].loc[data['image_id'] == item[:-4]] = [arr_im]

    with open('image_dataframe.pkl', 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    merge_im_array()


