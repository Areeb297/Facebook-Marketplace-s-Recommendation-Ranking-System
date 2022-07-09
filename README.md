# FB Marketplace Recommendation Ranking System

> This project includes building a ranking system to provide users with the most relevant products based on their search query through using deep learning frameworks like CNNs and Transfer Learning with the help of the Pytorch library.

## Milestone 1

- In this stage, we perform data cleaning steps for the product and image datasets where concerning the product tabular dataset, we ensure all null and duplicate values are removed, and the price column is converted to float and the product creation time on Facebook is converted to datetime. Regarding the images dataset, we create a pipeline which resizes all the images into one consistent format such that all images have the same number of channels and size. 

- We use the pandas library and Regex expression to clean the product dataset. For images, we use pillow and os libraries in python where the clean_image_data function takes in the path for folder containing all the images, opens all the images using a for loop, resizes all of them and saves them into a new directory called cleaned_images. Below is a snippet shown of the process of how we resize all images and having only RGB channels.
  
```python
size = im.size
ratio = float(final_size) / max(size) 
new_image_size = tuple([int(x*ratio) for x in size]) 
im = im.resize(new_image_size, Image.ANTIALIAS)
new_im = Image.new("RGB", (final_size, final_size))
new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
```


## Milestone 2

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
