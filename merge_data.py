import pandas as pd
from clean_tabular_data import get_data_pipeline, clean_time

def merge():
    """This function helps in merging the product and image datasets where every product can have more than one images

    Returns:
        pd.DataFrame: We return a cleaned dataframe ready to use for image classification
    
    """

    products_data = get_data_pipeline()
    images_data = pd.read_csv("images.csv")
    images_data.drop(columns='Unnamed: 0', inplace=True)

    images_data['create_time'] = clean_time(images_data['create_time'])

    df = pd.merge(left=products_data, right=images_data, on=('product_id', 'create_time'), sort=True)
    df.drop(columns=['product_id','create_time', 'bucket_link', 'image_ref'], inplace=True)

    return df

if __name__ == '__main__':
    merge()
