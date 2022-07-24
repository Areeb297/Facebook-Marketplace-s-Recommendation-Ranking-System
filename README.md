# FB Marketplace Recommendation Ranking System

## Milestone 1: An overview of the system

> This project works on building and replication a product ranking system of Facebook Marketplace to provide users with the most relevant products based on their search query through using using multimodal pre-traineddeep neural networks such as CNNs using Transfer Learning. The model is a mini implementation of a larger system that Facebook developed and advanced to generate product recommendation rankings for buyers on Facebook Marketplace. Shown below is a flowchart describing the overview of the system encompassing various technologies:

![image](https://user-images.githubusercontent.com/51030860/178149528-8a7c5b0c-3f14-46b0-b708-ff3faf455755.png)

Here is the ![video link](https://www.youtube.com/watch?v=1Z5V2VrHTTA&ab_channel=AiCore) for further information and reference.

## Milestone 2: Cleaning the tabular and image datasets

- In this stage, we perform data cleaning steps for the product and image datasets. Firstly, conncerning the product tabular dataset, we have a pipeline which completes all cleaning steps such as ensuring all null and duplicate values are removed and all data formats are correct e.g., the price column is converted to float and the product creation time is converted to datetime. Regarding the images dataset, we create a pipeline which resizes all the images into one consistent format such that all images have the same number of channels and size. An important point to mention is that the images in the folder are named by their id so we will sort these images first and then apply the image resizing pipeline. Similarly, when we merge both the image and product tabular datasets, we will sort the dataframe by image id which will help us in the classification task.

- Moreover, in order to have only the images that are in the image tabular dataset, first we merge the product and image datasets together, perform all the cleaning steps and then before resize the image, we check whether the id (name of the image file) is in the unique image id's in the tabular dataset. This ensures we have the same number of dimensions when performing image classification. Below is a code snippet that shows how we do it:

```python
# check if cleaned_images exists
new_path = 'cleaned_images/'
if not os.path.exists(path+new_path):
    os.makedirs(new_path)

final_size = 90

for n, item in enumerate(dirs, 1):
    if item.split('.')[0] in list(images_data['id'].unique()): # Here we check whether the image id is contained in the merged tabular dataset
        im = Image.open(path + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'{new_path}{n}_resized.jpg')
```

- We use the pandas library and Regex expression to clean the product dataset. For images, we use pillow and os libraries in python where the clean_image_data function takes in the path for the folder containing all the images, opens all the images using a for loop, resizes all of them and saves them into a new directory called cleaned_images. Below is a snippet shown of the process of how we resize all images and having only RGB channels. We only use the final size as 90 as large pixel sizes will increase the machine learning classification model time.
  
```python
size = im.size
ratio = float(final_size) / max(size) 
new_image_size = tuple([int(x*ratio) for x in size]) 
im = im.resize(new_image_size, Image.ANTIALIAS)
new_im = Image.new("RGB", (final_size, final_size))
new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
```


## Milestone 3: Create simple Machine Learning models

## Milestone 4: Create the vision model

## Milestone 5: Create the text understanding model

## Milestone 6: Combine the models

## Milestone 7: Configure and deploy the model serving API

- Does what you have built in this milestone connect to the previous one? If so explain how. What technologies are used? Why have you used them? Have you run any commands in the terminal? If so insert them using backticks (To get syntax highlighting for code snippets add the language after the first backticks).

- Example below:

```bash
/bin/kafka-topics.sh --list --zookeeper 127.0.0.1:2181
```

- The above command is used to check whether the topic has been created successfully, once confirmed the API script is edited to send data to the created kafka topic. The docker container has an attached volume which allows editing of files to persist on the container. The result of this is below:

```python
"""Insert your code here"""
```

> Insert screenshot of what you have built working.

## Milestone n

- Continue this process for every milestone, making sure to display clear understanding of each task and the concepts behind them as well as understanding of the technologies used.

- Also don't forget to include code snippets and screenshots of the system you are building, it gives proof as well as it being an easy way to evidence your experience!

## Conclusions

- Maybe write a conclusion to the project, what you understood about it and also how you would improve it or take it further.

- Read through your documentation, do you understand everything you've written? Is everything clear and cohesive?
