import pandas as pd
from clean_tabular_data import get_data_pipeline

def merge_data():
    """This function helps in merging the product and image datasets where every product can have more than one images

    Returns:
        pd.DataFrame: We return a cleaned dataframe ready to use for image classification
    
    """

    products_data = get_data_pipeline()
    images_data = pd.read_csv("images.csv").iloc[:, 1:]

    df = pd.merge(left=products_data, right=images_data, on=('product_id', 'create_time') , sort=True)
    df.drop(columns=['product_id', 'url', 'create_time', 'id', 'bucket_link', 'image_ref', 'page_id'], inplace=True)

    return df

