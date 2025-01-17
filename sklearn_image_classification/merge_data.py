import pandas as pd
import numpy as np
from clean_tabular_data import get_data_pipeline, clean_time
import os
from PIL import Image
import pickle
import seaborn as sns 
import matplotlib.pyplot as plt

def merge():
    """This function merges the product and image datasets where every product can have more than one image,
    keeps the first category from the category column converts the category column into numerical codes and lastly, 
    saves the mapping as a dictionary for decoding purposes 

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

    df.category = df.category.apply(lambda x: x.split('/')[0]) # Retain only the first category
    df.category =  df.category.astype('category')
    df['category_codes'] =  df.category.cat.codes # convert category column into numerical codes
    decoder_dict = dict(enumerate(df['category'].cat.categories))

    df.to_csv('product_images.csv') # This will be saved for later to use for deep learning/CNNs/transfer learning
    
    with open('decoder.pkl', 'wb') as file: # same encodings no matter how many times it is run
        pickle.dump(decoder_dict, file)

    return df

def generate_ml_classification_data(path, filename, show_plot=True):
    """This function gets the images from the cleaned_images folder, converts them to an array, merges them into the 
    main (merged) dataframe and saves it whilst also checking that the each image array corresponds to the correct image id

    Args:
        path (str): Path to where the image files are located
        filename (str): Name of pickle file where merged dataframe is going to be save
        show_plot (bool): If set true, the function will show the countplot of the categories
    """

    data = merge()
    dirs = os.listdir(path)
    if not os.path.exists(filename): # Check if one file exists already to see if we already have run this script    
        data['image_array'] = " " # initalize empty column in dataframe
        for item in dirs:
            if item[:-4] in data['image_id'].values:
                image = Image.open(path + item)
                arr_im = np.asarray(image)
                data['image_array'].loc[data['image_id'] == item[:-4]] = [arr_im] # assign the image array to the correct location/row in the dataframe

        with open(filename, 'wb') as file:
            pickle.dump(data, file) # Save the final dataframe as pickle file to keep image array format

    else:
        data = pd.read_pickle(filename)

    # Show that our target variabe has balanced classes and thus no need for oversampling/undersampling when using ML 
    if show_plot:
        sns.countplot(y=data.category)
        plt.show()


if __name__ == '__main__':
    generate_ml_classification_data('cleaned_images_ML/', 'ML_product_images.pkl') # 30x30 pixel size


