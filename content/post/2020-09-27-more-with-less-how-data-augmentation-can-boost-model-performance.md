---
title: More with Less - How Data Augmentation Can Boost Model Performance
author: Matt Cole
date: '2020-09-27'
slug: more-with-less-how-data-augmentation-can-boost-model-performance
categories:
  - Deep Learning
tags:
  - Data Augmentation
  - fast.ai
---
You may have heard that deep learning, particularly within image recognition situations, requires huge amounts of data. Mathematically, a ‘small’ [ResNet-18](https://en.wikipedia.org/wiki/Residual_neural_network) model has millions of trainable parameters, and as everyone knows from statistics class - we need more training data than model parameters, right?* While it remains true that, all else equal, the more data a model has access to in training the better it will perform - there are a few tricks we can use to increase the mileage of the data which we do have. As with life and deep learning, a greater exposure to different varieties and experiences helps us become better connoisseurs of those things. However, acquiring a large quantity (experiences) of labeled data with a sufficient diversity (varieties) is difficult, time consuming, and can even be expensive. If only we could generate more data from, well, the data we already had - our lives would be much easier.

### Enter, data augmentation.

Data Augmentation is the subtle altering of images (or more generally - data) prior to being fed to our model. This relatively simple procedure's objective is to manipulate the data being ingested by the model in such a way that the model will generalize better - ultimately requiring less data for similar levels of performance (another way of saying better performance with the same amount of data). 

Let’s see some data augmentation in action within a Tex-Mex food classification example using the super fun Fast.ai library.


```python
from fastai.vision.all import *
```

### Setting the scene:

Let's imagine ourselves as the chief developer for Silicon Valley's hottest new food start up Next-Mex, which we pitch to investors as 'kind of like Yelp - except only for Mexican food'. Our goal here is to be able to classify photos uploaded to the app/site as a particular type of Mexican food. Eventually we hope to identify trends within American's collective tex-mex palate and sell that data to Taco Bell for millions.


### The data

Well, the best thing to do here would be to get real images uploaded by the users of our site and classify them 'by hand' to create a training set for our Deep Learning model. Except we haven't launched yet (sorry [pg](http://www.paulgraham.com/13sentences.html))... 

Instead we grab some examples of several classic Tex-Mex foods from the Bing Image Search API. We have 150 examples of chimichangas, burritos, tacos, fajitas, and quesadillas respectively.

We can see 10 random examples of the data below.


```python
path = Path('mexican_foods')
```


```python
foods = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls = foods.dataloaders(path)
```


```python
dls = foods.dataloaders(path)
```


```python
dls.valid.show_batch(max_n=10, nrows=2)
```



![png](/post/2020-09-27-more-with-less-how-data-augmentation-can-boost-model-performance/output_6_0.png)
    


Let's look at one example in particular - an image of a quesadilla (resized to 128 px).


```python
foods_no_data_aug = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls_no_data_aug = foods_no_data_aug.dataloaders(path)
foods_no_data_aug = foods_no_data_aug.new(item_tfms=Resize(128))
dls_no_data_aug = foods_no_data_aug.dataloaders(path)
dls_no_data_aug.train.show_batch(max_n=1, nrows=1, unique=True)
```


    
![png](/post/2020-09-27-more-with-less-how-data-augmentation-can-boost-model-performance/output_8_0.png)
    



### Next steps:

What types of data augmentation would be useful here depends on the nature of the data (pictures of Mexican foods scooped up from Bing) and the purpose of the model (classifying user images of Mexican foods at Mexican restaurants).

We can imagine that users submitting pictures to Yelp may take photos at a variety of angles (some close, some far, some really zoomed in, etc.). In addition, the orientation of the photos horizontally (mirrored) doesn't matter, while we can probably assume that the users will probably not. We could also imagine that the pictures grabbed from Bing may include professionally done photographs taken with a steady hand (good lighting, no blur, proper angles, food is dead center, etc). These traits may not (let's be real - will not)  Fast.ai (the deep learning library I am using here) has a convenient function to do many data augmentation tasks aptly named `aug_transforms` which we will be using here. 


```python
foods_regular_transform = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls = foods_regular_transform.dataloaders(path)
foods_regular_transform = foods_regular_transform.new(item_tfms=Resize(128), 
                  batch_tfms=aug_transforms(mult=1.0, 
                                            do_flip=True, 
                                            flip_vert=False,
                                            max_rotate=45.0, 
                                            min_zoom=1.0, 
                                            max_zoom=1.0, 
                                            max_lighting=0, 
                                            max_warp=0, 
                                            p_affine=0.99, 
                                            p_lighting=1, 
                                            xtra_tfms=None, 
                                            size=None, 
                                            mode='bilinear', 
                                            pad_mode='reflection', 
                                            align_corners=True, 
                                            batch=False, 
                                            min_scale=1.0))
dls = foods_regular_transform.dataloaders(path)
dls.train.show_batch(max_n=5, nrows=1, unique=True)
```


    
![png](/post/2020-09-27-more-with-less-how-data-augmentation-can-boost-model-performance/output_10_0.png)
    


Now, on each Epoch the data loader will pass on a random 'version' of this image that will be rotated randomly up to 45 degrees and will have a 50% probability of being rotated along the vertical axis. 

Our model will now be trained on non-perfect images which will help it classify foods based on characteristics of the foods themselves rather than non-loyal artifacts of the training data. To conceptualize this we can imagine a car make and model classifier.

If a lot of Prius' were shown perpendicular our model may 'learn' that all (or most) images with strong, prominent horizontal edges are actually a Prius. This is an artifact of the data, not of the actual thing (Prius) that we are trying to model.
![Perpendicular Car](/post/2020-09-27-more-with-less-how-data-augmentation-can-boost-model-performance/car.jpeg)


### More augmentation

There are of course, many other types of data augmentation. In fact, nearly anything you can think of (adding green hues, increasing image distortion, making darker, removing portions of the image etc.) can be a valid approach to data augmentation in the correct setting. 

In this case, we will add some perspective warping (arg: `max_warp`), mirroring (arg `do_flip`), rotating (arg: `max_rotate`), zooming in (args: `in_zoom` + `max_zoom`) which are all conveniently options in the incredibly useful `aug_transforms` function. We can set `mult` > 1 for exaggerated effects so we can see the results more easily (note: too much augmentation could in fact be detrimental as the images can become cartoonish and not represent the actual images which we will be conducing inference on).


```python
foods_regular_transform = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls = foods_regular_transform.dataloaders(path)
foods_regular_transform = foods_regular_transform.new(item_tfms=Resize(128), 
                  batch_tfms=aug_transforms(mult=2.0, 
                                            do_flip=True, 
                                            flip_vert=False,
                                            max_rotate=10.0, 
                                            min_zoom=1.0, 
                                            max_zoom=1.2, 
                                            max_lighting=0.3, 
                                            max_warp=0.2, 
                                            p_affine=0.75, 
                                            p_lighting=0.9, 
                                            xtra_tfms=None, 
                                            size=None, 
                                            mode='bilinear', 
                                            pad_mode='reflection', 
                                            align_corners=True, 
                                            batch=False, min_scale=1.0))
dls = foods_regular_transform.dataloaders(path)
dls.train.show_batch(max_n=10, nrows=2, unique=True)
```


    
![png](/post/2020-09-27-more-with-less-how-data-augmentation-can-boost-model-performance/output_13_0.png)
    


What I really love about these images is they really do look like different images of quesadillas. Some are darker, some are lighter, it appears as if we have a whole bunch of vantage points (thanks to the perspective warping which I love!). These synthetically generated images will allow our model to better generalize to new images of Mexican foods as the style of the image changed, but they all remain qualitatively quesadillas. 

Thanks for reading! Let me know if you have any comments or questions below!

  

\*This statement is touching upon the phenomenon that neural networks tend to be surprisingly robust to overfitting. The reasons why include Dropout, Architecture selection, regularization, and more data + data augmentation if applicable.



