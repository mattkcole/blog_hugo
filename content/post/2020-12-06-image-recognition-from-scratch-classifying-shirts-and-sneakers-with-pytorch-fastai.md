---
title: Computer Vision From Scratch - Classifying Shirts and Sneakers with Pytorch
  & FastAI
author: Matt Cole
date: '2020-12-06'
slug: image-classification-from-scratch
categories:
  - Deep Learning
tags:
  - Pytorch
  - deep learning
---

This post will take us through the process of building, from scratch, an image classification model using stochastic gradient descent (SGD).

Ok, great - building a model from scratch. But what should we model?
  
Since the 90's the default data set for testing out image classification models has been [MNIST](https://en.wikipedia.org/wiki/MNIST_database) - a collection of 70,000 greyscale handwritten digits. For the most part it's a great data set, the major drawback (in my opinion) - it's incredibly boring (maybe if they used my handwriting it would be more exciting?).

![MNIST](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

 
\[Above\] A visualization of some of the MNIST data taken from Wikipedia.


Thankfully, a team of researchers put together [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), a fun collection of greyscale fashion images.

Similarly to MNIST there are 10 classes, but each is a type of clothing.

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |



Examples of these data below (from the project Github page - each class takes three rows):
![Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png?raw=true)


Well, I may be fashion challenged  - but the fashion MNIST is much more entertaining - let's use that for this experiment!

# Project Goal

In this lil' post, we will train a binary classifier to determine if a given image is a sneaker or a shirt. 

## Getting Started

To start, let's load the important libraries and explore our data.


```python
from fastai.vision.all import *
import numpy as np
import torchvision
import torchvision.transforms as transforms
%matplotlib inline
import matplotlib.pyplot as plt
matplotlib.rc('image', cmap='Greys')
```

## Getting the data

Pytorch conveniently has some [shortcuts](https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist) for us to download the Fashion MNIST data, which we'll take advantage of here. And, like most curated ML data sets, there are pre-defined training and test sets.


```python
trainset = torchvision.datasets.FashionMNIST(root = "./notebooks/storage", train = True, download = True, transform = transforms.ToTensor())
testset = torchvision.datasets.FashionMNIST(root = "./notebooks/storage", train = False, download = True, transform = transforms.ToTensor())
```

Ok, what does this data look like?


```python
type(trainset)
```




    torchvision.datasets.mnist.FashionMNIST




```python
print(trainset.data.shape, trainset.targets.shape)
```

    torch.Size([60000, 28, 28]) torch.Size([60000])


The data sets contain primarily a `.data` and `.targets` attributes. `.data` is a rank 3 tensor of shape [60000, 28, 28]. In reality, this tensor is 60,000 images of size 28px by 28px. As expected, `.targets` is a rank 1 tensor of shape 60,000 - indicating the class of each image in `.data`.

Since in this project we are only interested in two types of clothing, let's grab our sneakers (7) and shirts (6). 


```python
trainset_x = trainset.data[(trainset.targets == 6) | (trainset.targets == 7)]
trainset_y = trainset.targets[(trainset.targets == 6) | (trainset.targets == 7)]
```

Next, we can split our trainset into training and validation sets (remember - the test set was already broken out for us).


```python
d_length = trainset_x.shape[0]
# 10% can go to validation
validation_ix = random.sample(range(0, d_length),round(d_length*0.1))
train_ix = np.setdiff1d(range(0, d_length),validation_ix)
```


```python
trainX = trainset_x[train_ix]
trainy  = trainset_y[train_ix]
validX = trainset_x[validation_ix]
validy = trainset_y[validation_ix]
print(f'training set size: {trainy.shape[0]}\nvalidation set size: {validy.shape[0]}')
```

    training set size: 10800
    validation set size: 1200


We can take a peak with FastAI's handy `show_image` function to see what this data looks like. 


```python
random_ix = [0,100,2]
show_image(trainX[random_ix[0]]); 
show_image(trainX[random_ix[1]]); 
show_image(trainX[random_ix[2]])
print(['shirt' if i.item() == 6 else 'sneaker' for i in trainy[random_ix]])
```

    ['sneaker', 'sneaker', 'shirt']



    
![png](/post/2020-12-06-image-classification-from-scratch_files/output_19_1.png)
    



    
![png](/post/2020-12-06-image-classification-from-scratch_files/output_19_2.png)
    



    
![png](/post/2020-12-06-image-classification-from-scratch_files/output_19_3.png)
    


Ok, those pictures are pretty cool, but what does that data actually look like? Surely the data has some sort of numerical representation?

Well, we already know each image is a 28x28 tensor. Each element of that matrix is an integer representing its darkness (0 is white, 255 is black, everything in the middle is grey).

Let's take a quick look at a pandas heatmap representation of the data. We can see the 28X28 grid, the values ranging from 0 to 255.


```python
df = pd.DataFrame(trainX[0])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```




<style  type="text/css" >
#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col27{
            font-size:  6pt;
            background-color:  #ffffff;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col24{
            font-size:  6pt;
            background-color:  #fefefe;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col12{
            font-size:  6pt;
            background-color:  #fdfdfd;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col18{
            font-size:  6pt;
            background-color:  #aaaaaa;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col19{
            font-size:  6pt;
            background-color:  #030303;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col15{
            font-size:  6pt;
            background-color:  #c1c1c1;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col25{
            font-size:  6pt;
            background-color:  #8a8a8a;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col11{
            font-size:  6pt;
            background-color:  #252525;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col23{
            font-size:  6pt;
            background-color:  #232323;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col7{
            font-size:  6pt;
            background-color:  #1d1d1d;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col20{
            font-size:  6pt;
            background-color:  #616161;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col5{
            font-size:  6pt;
            background-color:  #fcfcfc;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col23{
            font-size:  6pt;
            background-color:  #ebebeb;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col13{
            font-size:  6pt;
            background-color:  #959595;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col14{
            font-size:  6pt;
            background-color:  #5a5a5a;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col21{
            font-size:  6pt;
            background-color:  #050505;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col8{
            font-size:  6pt;
            background-color:  #323232;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col18{
            font-size:  6pt;
            background-color:  #4d4d4d;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col18{
            font-size:  6pt;
            background-color:  #535353;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col10{
            font-size:  6pt;
            background-color:  #171717;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col19{
            font-size:  6pt;
            background-color:  #000000;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col24{
            font-size:  6pt;
            background-color:  #bebebe;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col13{
            font-size:  6pt;
            background-color:  #474747;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col6{
            font-size:  6pt;
            background-color:  #f7f7f7;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col11{
            font-size:  6pt;
            background-color:  #767676;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col13{
            font-size:  6pt;
            background-color:  #141414;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col12{
            font-size:  6pt;
            background-color:  #3d3d3d;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col11{
            font-size:  6pt;
            background-color:  #4a4a4a;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col3{
            font-size:  6pt;
            background-color:  #464646;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col16{
            font-size:  6pt;
            background-color:  #404040;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col18{
            font-size:  6pt;
            background-color:  #6f6f6f;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col20{
            font-size:  6pt;
            background-color:  #9c9c9c;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col7{
            font-size:  6pt;
            background-color:  #f4f4f4;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col27{
            font-size:  6pt;
            background-color:  #a5a5a5;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col21{
            font-size:  6pt;
            background-color:  #ececec;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col18{
            font-size:  6pt;
            background-color:  #dcdcdc;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col19{
            font-size:  6pt;
            background-color:  #d2d2d2;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col9{
            font-size:  6pt;
            background-color:  #636363;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col9{
            font-size:  6pt;
            background-color:  #1a1a1a;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col19{
            font-size:  6pt;
            background-color:  #393939;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col10{
            font-size:  6pt;
            background-color:  #2f2f2f;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col16{
            font-size:  6pt;
            background-color:  #3f3f3f;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col24{
            font-size:  6pt;
            background-color:  #2c2c2c;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col7{
            font-size:  6pt;
            background-color:  #070707;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col23{
            font-size:  6pt;
            background-color:  #545454;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col25,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col5{
            font-size:  6pt;
            background-color:  #fafafa;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col20{
            font-size:  6pt;
            background-color:  #f3f3f3;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col6{
            font-size:  6pt;
            background-color:  #939393;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col9{
            font-size:  6pt;
            background-color:  #0d0d0d;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col12{
            font-size:  6pt;
            background-color:  #414141;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col6,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col19{
            font-size:  6pt;
            background-color:  #1e1e1e;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col13,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col7,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col20{
            font-size:  6pt;
            background-color:  #222222;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col15{
            font-size:  6pt;
            background-color:  #303030;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col16{
            font-size:  6pt;
            background-color:  #3c3c3c;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col17{
            font-size:  6pt;
            background-color:  #505050;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col19,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col4,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col24{
            font-size:  6pt;
            background-color:  #2b2b2b;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col21{
            font-size:  6pt;
            background-color:  #515151;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col24{
            font-size:  6pt;
            background-color:  #989898;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col5{
            font-size:  6pt;
            background-color:  #151515;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col27,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col1{
            font-size:  6pt;
            background-color:  #d6d6d6;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col11{
            font-size:  6pt;
            background-color:  #7a7a7a;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col20,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col14{
            font-size:  6pt;
            background-color:  #434343;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col6{
            font-size:  6pt;
            background-color:  #101010;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col10{
            font-size:  6pt;
            background-color:  #444444;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col9,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col21{
            font-size:  6pt;
            background-color:  #161616;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col12{
            font-size:  6pt;
            background-color:  #1f1f1f;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col15{
            font-size:  6pt;
            background-color:  #353535;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col26,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col20{
            font-size:  6pt;
            background-color:  #111111;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col5{
            font-size:  6pt;
            background-color:  #0a0a0a;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col18{
            font-size:  6pt;
            background-color:  #383838;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col24{
            font-size:  6pt;
            background-color:  #333333;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col25{
            font-size:  6pt;
            background-color:  #242424;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col0,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col4{
            font-size:  6pt;
            background-color:  #eeeeee;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col1{
            font-size:  6pt;
            background-color:  #d0d0d0;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col2{
            font-size:  6pt;
            background-color:  #828282;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col4{
            font-size:  6pt;
            background-color:  #080808;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col8{
            font-size:  6pt;
            background-color:  #181818;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col21{
            font-size:  6pt;
            background-color:  #0e0e0e;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col22{
            font-size:  6pt;
            background-color:  #0c0c0c;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col25{
            font-size:  6pt;
            background-color:  #292929;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col26{
            font-size:  6pt;
            background-color:  #090909;
            color:  #f1f1f1;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col1,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col6{
            font-size:  6pt;
            background-color:  #f8f8f8;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col4{
            font-size:  6pt;
            background-color:  #fbfbfb;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col17,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col17{
            font-size:  6pt;
            background-color:  #e4e4e4;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col6{
            font-size:  6pt;
            background-color:  #c2c2c2;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col7{
            font-size:  6pt;
            background-color:  #999999;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col8{
            font-size:  6pt;
            background-color:  #848484;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col9{
            font-size:  6pt;
            background-color:  #6d6d6d;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col12{
            font-size:  6pt;
            background-color:  #7b7b7b;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col13{
            font-size:  6pt;
            background-color:  #949494;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col14{
            font-size:  6pt;
            background-color:  #a7a7a7;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col15{
            font-size:  6pt;
            background-color:  #bfbfbf;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col16{
            font-size:  6pt;
            background-color:  #cecece;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col17{
            font-size:  6pt;
            background-color:  #dbdbdb;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col18{
            font-size:  6pt;
            background-color:  #e5e5e5;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col21,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col5,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col13{
            font-size:  6pt;
            background-color:  #f2f2f2;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col23,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col3,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col19{
            font-size:  6pt;
            background-color:  #efefef;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col24,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col2,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col20{
            font-size:  6pt;
            background-color:  #f0f0f0;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col25{
            font-size:  6pt;
            background-color:  #f9f9f9;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col7{
            font-size:  6pt;
            background-color:  #f6f6f6;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col8,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col26{
            font-size:  6pt;
            background-color:  #f1f1f1;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col15{
            font-size:  6pt;
            background-color:  #ededed;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col16,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col10,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col11,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col14,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col15,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col25{
            font-size:  6pt;
            background-color:  #e9e9e9;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col20{
            font-size:  6pt;
            background-color:  #c8c8c8;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col21{
            font-size:  6pt;
            background-color:  #c0c0c0;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col22,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col23{
            font-size:  6pt;
            background-color:  #c5c5c5;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col25{
            font-size:  6pt;
            background-color:  #c3c3c3;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col26{
            font-size:  6pt;
            background-color:  #c4c4c4;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col27{
            font-size:  6pt;
            background-color:  #c7c7c7;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col12,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col18,#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col22{
            font-size:  6pt;
            background-color:  #e7e7e7;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col13{
            font-size:  6pt;
            background-color:  #e6e6e6;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col16{
            font-size:  6pt;
            background-color:  #e8e8e8;
            color:  #000000;
        }#T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col24{
            font-size:  6pt;
            background-color:  #eaeaea;
            color:  #000000;
        }</style><table id="T_08e7b92e_3844_11eb_9c0a_0242ac110002" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>        <th class="col_heading level0 col18" >18</th>        <th class="col_heading level0 col19" >19</th>        <th class="col_heading level0 col20" >20</th>        <th class="col_heading level0 col21" >21</th>        <th class="col_heading level0 col22" >22</th>        <th class="col_heading level0 col23" >23</th>        <th class="col_heading level0 col24" >24</th>        <th class="col_heading level0 col25" >25</th>        <th class="col_heading level0 col26" >26</th>        <th class="col_heading level0 col27" >27</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col0" class="data row0 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col1" class="data row0 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col2" class="data row0 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col3" class="data row0 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col4" class="data row0 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col5" class="data row0 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col6" class="data row0 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col7" class="data row0 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col8" class="data row0 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col9" class="data row0 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col10" class="data row0 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col11" class="data row0 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col12" class="data row0 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col13" class="data row0 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col14" class="data row0 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col15" class="data row0 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col16" class="data row0 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col17" class="data row0 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col18" class="data row0 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col19" class="data row0 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col20" class="data row0 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col21" class="data row0 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col22" class="data row0 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col23" class="data row0 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col24" class="data row0 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col25" class="data row0 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col26" class="data row0 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row0_col27" class="data row0 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col0" class="data row1 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col1" class="data row1 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col2" class="data row1 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col3" class="data row1 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col4" class="data row1 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col5" class="data row1 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col6" class="data row1 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col7" class="data row1 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col8" class="data row1 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col9" class="data row1 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col10" class="data row1 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col11" class="data row1 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col12" class="data row1 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col13" class="data row1 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col14" class="data row1 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col15" class="data row1 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col16" class="data row1 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col17" class="data row1 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col18" class="data row1 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col19" class="data row1 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col20" class="data row1 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col21" class="data row1 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col22" class="data row1 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col23" class="data row1 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col24" class="data row1 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col25" class="data row1 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col26" class="data row1 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row1_col27" class="data row1 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col0" class="data row2 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col1" class="data row2 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col2" class="data row2 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col3" class="data row2 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col4" class="data row2 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col5" class="data row2 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col6" class="data row2 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col7" class="data row2 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col8" class="data row2 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col9" class="data row2 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col10" class="data row2 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col11" class="data row2 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col12" class="data row2 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col13" class="data row2 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col14" class="data row2 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col15" class="data row2 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col16" class="data row2 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col17" class="data row2 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col18" class="data row2 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col19" class="data row2 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col20" class="data row2 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col21" class="data row2 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col22" class="data row2 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col23" class="data row2 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col24" class="data row2 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col25" class="data row2 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col26" class="data row2 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row2_col27" class="data row2 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col0" class="data row3 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col1" class="data row3 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col2" class="data row3 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col3" class="data row3 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col4" class="data row3 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col5" class="data row3 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col6" class="data row3 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col7" class="data row3 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col8" class="data row3 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col9" class="data row3 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col10" class="data row3 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col11" class="data row3 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col12" class="data row3 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col13" class="data row3 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col14" class="data row3 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col15" class="data row3 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col16" class="data row3 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col17" class="data row3 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col18" class="data row3 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col19" class="data row3 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col20" class="data row3 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col21" class="data row3 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col22" class="data row3 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col23" class="data row3 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col24" class="data row3 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col25" class="data row3 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col26" class="data row3 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row3_col27" class="data row3 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col0" class="data row4 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col1" class="data row4 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col2" class="data row4 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col3" class="data row4 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col4" class="data row4 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col5" class="data row4 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col6" class="data row4 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col7" class="data row4 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col8" class="data row4 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col9" class="data row4 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col10" class="data row4 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col11" class="data row4 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col12" class="data row4 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col13" class="data row4 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col14" class="data row4 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col15" class="data row4 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col16" class="data row4 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col17" class="data row4 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col18" class="data row4 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col19" class="data row4 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col20" class="data row4 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col21" class="data row4 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col22" class="data row4 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col23" class="data row4 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col24" class="data row4 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col25" class="data row4 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col26" class="data row4 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row4_col27" class="data row4 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col0" class="data row5 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col1" class="data row5 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col2" class="data row5 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col3" class="data row5 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col4" class="data row5 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col5" class="data row5 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col6" class="data row5 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col7" class="data row5 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col8" class="data row5 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col9" class="data row5 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col10" class="data row5 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col11" class="data row5 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col12" class="data row5 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col13" class="data row5 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col14" class="data row5 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col15" class="data row5 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col16" class="data row5 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col17" class="data row5 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col18" class="data row5 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col19" class="data row5 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col20" class="data row5 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col21" class="data row5 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col22" class="data row5 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col23" class="data row5 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col24" class="data row5 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col25" class="data row5 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col26" class="data row5 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row5_col27" class="data row5 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col0" class="data row6 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col1" class="data row6 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col2" class="data row6 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col3" class="data row6 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col4" class="data row6 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col5" class="data row6 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col6" class="data row6 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col7" class="data row6 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col8" class="data row6 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col9" class="data row6 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col10" class="data row6 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col11" class="data row6 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col12" class="data row6 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col13" class="data row6 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col14" class="data row6 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col15" class="data row6 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col16" class="data row6 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col17" class="data row6 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col18" class="data row6 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col19" class="data row6 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col20" class="data row6 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col21" class="data row6 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col22" class="data row6 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col23" class="data row6 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col24" class="data row6 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col25" class="data row6 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col26" class="data row6 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row6_col27" class="data row6 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col0" class="data row7 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col1" class="data row7 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col2" class="data row7 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col3" class="data row7 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col4" class="data row7 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col5" class="data row7 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col6" class="data row7 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col7" class="data row7 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col8" class="data row7 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col9" class="data row7 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col10" class="data row7 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col11" class="data row7 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col12" class="data row7 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col13" class="data row7 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col14" class="data row7 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col15" class="data row7 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col16" class="data row7 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col17" class="data row7 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col18" class="data row7 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col19" class="data row7 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col20" class="data row7 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col21" class="data row7 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col22" class="data row7 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col23" class="data row7 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col24" class="data row7 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col25" class="data row7 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col26" class="data row7 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row7_col27" class="data row7 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col0" class="data row8 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col1" class="data row8 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col2" class="data row8 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col3" class="data row8 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col4" class="data row8 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col5" class="data row8 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col6" class="data row8 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col7" class="data row8 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col8" class="data row8 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col9" class="data row8 col9" >1</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col10" class="data row8 col10" >1</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col11" class="data row8 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col12" class="data row8 col12" >3</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col13" class="data row8 col13" >1</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col14" class="data row8 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col15" class="data row8 col15" >4</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col16" class="data row8 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col17" class="data row8 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col18" class="data row8 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col19" class="data row8 col19" >2</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col20" class="data row8 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col21" class="data row8 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col22" class="data row8 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col23" class="data row8 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col24" class="data row8 col24" >5</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col25" class="data row8 col25" >1</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col26" class="data row8 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row8_col27" class="data row8 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col0" class="data row9 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col1" class="data row9 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col2" class="data row9 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col3" class="data row9 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col4" class="data row9 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col5" class="data row9 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col6" class="data row9 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col7" class="data row9 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col8" class="data row9 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col9" class="data row9 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col10" class="data row9 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col11" class="data row9 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col12" class="data row9 col12" >4</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col13" class="data row9 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col14" class="data row9 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col15" class="data row9 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col16" class="data row9 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col17" class="data row9 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col18" class="data row9 col18" >106</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col19" class="data row9 col19" >229</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col20" class="data row9 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col21" class="data row9 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col22" class="data row9 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col23" class="data row9 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col24" class="data row9 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col25" class="data row9 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col26" class="data row9 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row9_col27" class="data row9 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col0" class="data row10 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col1" class="data row10 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col2" class="data row10 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col3" class="data row10 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col4" class="data row10 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col5" class="data row10 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col6" class="data row10 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col7" class="data row10 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col8" class="data row10 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col9" class="data row10 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col10" class="data row10 col10" >1</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col11" class="data row10 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col12" class="data row10 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col13" class="data row10 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col14" class="data row10 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col15" class="data row10 col15" >90</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col16" class="data row10 col16" >138</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col17" class="data row10 col17" >223</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col18" class="data row10 col18" >214</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col19" class="data row10 col19" >209</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col20" class="data row10 col20" >167</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col21" class="data row10 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col22" class="data row10 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col23" class="data row10 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col24" class="data row10 col24" >6</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col25" class="data row10 col25" >124</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col26" class="data row10 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row10_col27" class="data row10 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col0" class="data row11 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col1" class="data row11 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col2" class="data row11 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col3" class="data row11 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col4" class="data row11 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col5" class="data row11 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col6" class="data row11 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col7" class="data row11 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col8" class="data row11 col8" >1</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col9" class="data row11 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col10" class="data row11 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col11" class="data row11 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col12" class="data row11 col12" >37</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col13" class="data row11 col13" >122</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col14" class="data row11 col14" >179</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col15" class="data row11 col15" >249</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col16" class="data row11 col16" >214</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col17" class="data row11 col17" >195</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col18" class="data row11 col18" >181</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col19" class="data row11 col19" >213</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col20" class="data row11 col20" >241</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col21" class="data row11 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col22" class="data row11 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col23" class="data row11 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col24" class="data row11 col24" >94</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col25" class="data row11 col25" >179</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col26" class="data row11 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row11_col27" class="data row11 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col0" class="data row12 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col1" class="data row12 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col2" class="data row12 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col3" class="data row12 col3" >2</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col4" class="data row12 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col5" class="data row12 col5" >6</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col6" class="data row12 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col7" class="data row12 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col8" class="data row12 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col9" class="data row12 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col10" class="data row12 col10" >16</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col11" class="data row12 col11" >149</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col12" class="data row12 col12" >236</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col13" class="data row12 col13" >226</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col14" class="data row12 col14" >201</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col15" class="data row12 col15" >195</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col16" class="data row12 col16" >200</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col17" class="data row12 col17" >204</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col18" class="data row12 col18" >155</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col19" class="data row12 col19" >209</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col20" class="data row12 col20" >116</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col21" class="data row12 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col22" class="data row12 col22" >22</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col23" class="data row12 col23" >109</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col24" class="data row12 col24" >251</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col25" class="data row12 col25" >35</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col26" class="data row12 col26" >51</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row12_col27" class="data row12 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col0" class="data row13 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col1" class="data row13 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col2" class="data row13 col2" >1</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col3" class="data row13 col3" >3</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col4" class="data row13 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col5" class="data row13 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col6" class="data row13 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col7" class="data row13 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col8" class="data row13 col8" >67</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col9" class="data row13 col9" >150</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col10" class="data row13 col10" >240</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col11" class="data row13 col11" >221</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col12" class="data row13 col12" >194</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col13" class="data row13 col13" >190</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col14" class="data row13 col14" >204</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col15" class="data row13 col15" >214</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col16" class="data row13 col16" >205</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col17" class="data row13 col17" >195</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col18" class="data row13 col18" >207</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col19" class="data row13 col19" >185</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col20" class="data row13 col20" >206</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col21" class="data row13 col21" >233</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col22" class="data row13 col22" >224</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col23" class="data row13 col23" >179</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col24" class="data row13 col24" >2</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col25" class="data row13 col25" >10</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col26" class="data row13 col26" >22</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row13_col27" class="data row13 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col0" class="data row14 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col1" class="data row14 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col2" class="data row14 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col3" class="data row14 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col4" class="data row14 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col5" class="data row14 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col6" class="data row14 col6" >110</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col7" class="data row14 col7" >214</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col8" class="data row14 col8" >237</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col9" class="data row14 col9" >209</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col10" class="data row14 col10" >196</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col11" class="data row14 col11" >192</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col12" class="data row14 col12" >215</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col13" class="data row14 col13" >215</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col14" class="data row14 col14" >213</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col15" class="data row14 col15" >213</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col16" class="data row14 col16" >207</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col17" class="data row14 col17" >193</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col18" class="data row14 col18" >186</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col19" class="data row14 col19" >199</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col20" class="data row14 col20" >206</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col21" class="data row14 col21" >175</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col22" class="data row14 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col23" class="data row14 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col24" class="data row14 col24" >124</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col25" class="data row14 col25" >230</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col26" class="data row14 col26" >200</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row14_col27" class="data row14 col27" >36</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col0" class="data row15 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col1" class="data row15 col1" >50</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col2" class="data row15 col2" >119</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col3" class="data row15 col3" >158</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col4" class="data row15 col4" >166</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col5" class="data row15 col5" >192</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col6" class="data row15 col6" >204</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col7" class="data row15 col7" >198</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col8" class="data row15 col8" >187</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col9" class="data row15 col9" >202</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col10" class="data row15 col10" >203</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col11" class="data row15 col11" >211</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col12" class="data row15 col12" >214</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col13" class="data row15 col13" >204</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col14" class="data row15 col14" >209</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col15" class="data row15 col15" >210</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col16" class="data row15 col16" >204</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col17" class="data row15 col17" >197</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col18" class="data row15 col18" >191</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col19" class="data row15 col19" >190</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col20" class="data row15 col20" >191</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col21" class="data row15 col21" >229</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col22" class="data row15 col22" >230</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col23" class="data row15 col23" >242</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col24" class="data row15 col24" >214</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col25" class="data row15 col25" >193</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col26" class="data row15 col26" >203</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row15_col27" class="data row15 col27" >137</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col0" class="data row16 col0" >108</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col1" class="data row16 col1" >190</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col2" class="data row16 col2" >199</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col3" class="data row16 col3" >200</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col4" class="data row16 col4" >194</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col5" class="data row16 col5" >199</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col6" class="data row16 col6" >194</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col7" class="data row16 col7" >195</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col8" class="data row16 col8" >199</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col9" class="data row16 col9" >200</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col10" class="data row16 col10" >189</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col11" class="data row16 col11" >187</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col12" class="data row16 col12" >191</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col13" class="data row16 col13" >189</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col14" class="data row16 col14" >197</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col15" class="data row16 col15" >198</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col16" class="data row16 col16" >205</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col17" class="data row16 col17" >200</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col18" class="data row16 col18" >200</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col19" class="data row16 col19" >208</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col20" class="data row16 col20" >213</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col21" class="data row16 col21" >215</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col22" class="data row16 col22" >212</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col23" class="data row16 col23" >213</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col24" class="data row16 col24" >209</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col25" class="data row16 col25" >202</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col26" class="data row16 col26" >216</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row16_col27" class="data row16 col27" >137</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col0" class="data row17 col0" >15</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col1" class="data row17 col1" >55</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col2" class="data row17 col2" >114</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col3" class="data row17 col3" >157</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col4" class="data row17 col4" >188</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col5" class="data row17 col5" >207</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col6" class="data row17 col6" >216</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col7" class="data row17 col7" >220</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col8" class="data row17 col8" >217</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col9" class="data row17 col9" >219</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col10" class="data row17 col10" >221</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col11" class="data row17 col11" >242</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col12" class="data row17 col12" >240</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col13" class="data row17 col13" >243</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col14" class="data row17 col14" >249</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col15" class="data row17 col15" >253</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col16" class="data row17 col16" >255</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col17" class="data row17 col17" >255</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col18" class="data row17 col18" >243</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col19" class="data row17 col19" >232</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col20" class="data row17 col20" >226</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col21" class="data row17 col21" >222</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col22" class="data row17 col22" >221</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col23" class="data row17 col23" >213</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col24" class="data row17 col24" >215</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col25" class="data row17 col25" >198</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col26" class="data row17 col26" >209</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row17_col27" class="data row17 col27" >62</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col0" class="data row18 col0" >16</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col1" class="data row18 col1" >11</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col2" class="data row18 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col3" class="data row18 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col4" class="data row18 col4" >7</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col5" class="data row18 col5" >40</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col6" class="data row18 col6" >76</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col7" class="data row18 col7" >108</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col8" class="data row18 col8" >134</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col9" class="data row18 col9" >142</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col10" class="data row18 col10" >143</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col11" class="data row18 col11" >145</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col12" class="data row18 col12" >143</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col13" class="data row18 col13" >123</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col14" class="data row18 col14" >111</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col15" class="data row18 col15" >92</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col16" class="data row18 col16" >76</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col17" class="data row18 col17" >61</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col18" class="data row18 col18" >45</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col19" class="data row18 col19" >35</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col20" class="data row18 col20" >25</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col21" class="data row18 col21" >25</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col22" class="data row18 col22" >31</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col23" class="data row18 col23" >32</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col24" class="data row18 col24" >32</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col25" class="data row18 col25" >12</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col26" class="data row18 col26" >1</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row18_col27" class="data row18 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col0" class="data row19 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col1" class="data row19 col1" >11</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col2" class="data row19 col2" >25</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col3" class="data row19 col3" >26</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col4" class="data row19 col4" >26</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col5" class="data row19 col5" >22</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col6" class="data row19 col6" >12</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col7" class="data row19 col7" >20</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col8" class="data row19 col8" >15</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col9" class="data row19 col9" >15</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col10" class="data row19 col10" >18</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col11" class="data row19 col11" >17</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col12" class="data row19 col12" >19</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col13" class="data row19 col13" >27</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col14" class="data row19 col14" >30</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col15" class="data row19 col15" >36</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col16" class="data row19 col16" >41</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col17" class="data row19 col17" >49</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col18" class="data row19 col18" >57</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col19" class="data row19 col19" >66</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col20" class="data row19 col20" >79</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col21" class="data row19 col21" >84</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col22" class="data row19 col22" >79</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col23" class="data row19 col23" >83</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col24" class="data row19 col24" >93</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col25" class="data row19 col25" >80</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col26" class="data row19 col26" >75</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row19_col27" class="data row19 col27" >45</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col0" class="data row20 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col1" class="data row20 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col2" class="data row20 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col3" class="data row20 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col4" class="data row20 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col5" class="data row20 col5" >9</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col6" class="data row20 col6" >14</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col7" class="data row20 col7" >17</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col8" class="data row20 col8" >27</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col9" class="data row20 col9" >34</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col10" class="data row20 col10" >39</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col11" class="data row20 col11" >39</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col12" class="data row20 col12" >42</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col13" class="data row20 col13" >44</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col14" class="data row20 col14" >41</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col15" class="data row20 col15" >41</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col16" class="data row20 col16" >43</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col17" class="data row20 col17" >48</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col18" class="data row20 col18" >43</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col19" class="data row20 col19" >30</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col20" class="data row20 col20" >31</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col21" class="data row20 col21" >35</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col22" class="data row20 col22" >40</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col23" class="data row20 col23" >37</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col24" class="data row20 col24" >40</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col25" class="data row20 col25" >37</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col26" class="data row20 col26" >26</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row20_col27" class="data row20 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col0" class="data row21 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col1" class="data row21 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col2" class="data row21 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col3" class="data row21 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col4" class="data row21 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col5" class="data row21 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col6" class="data row21 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col7" class="data row21 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col8" class="data row21 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col9" class="data row21 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col10" class="data row21 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col11" class="data row21 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col12" class="data row21 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col13" class="data row21 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col14" class="data row21 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col15" class="data row21 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col16" class="data row21 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col17" class="data row21 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col18" class="data row21 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col19" class="data row21 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col20" class="data row21 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col21" class="data row21 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col22" class="data row21 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col23" class="data row21 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col24" class="data row21 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col25" class="data row21 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col26" class="data row21 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row21_col27" class="data row21 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col0" class="data row22 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col1" class="data row22 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col2" class="data row22 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col3" class="data row22 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col4" class="data row22 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col5" class="data row22 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col6" class="data row22 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col7" class="data row22 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col8" class="data row22 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col9" class="data row22 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col10" class="data row22 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col11" class="data row22 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col12" class="data row22 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col13" class="data row22 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col14" class="data row22 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col15" class="data row22 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col16" class="data row22 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col17" class="data row22 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col18" class="data row22 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col19" class="data row22 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col20" class="data row22 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col21" class="data row22 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col22" class="data row22 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col23" class="data row22 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col24" class="data row22 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col25" class="data row22 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col26" class="data row22 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row22_col27" class="data row22 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col0" class="data row23 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col1" class="data row23 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col2" class="data row23 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col3" class="data row23 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col4" class="data row23 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col5" class="data row23 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col6" class="data row23 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col7" class="data row23 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col8" class="data row23 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col9" class="data row23 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col10" class="data row23 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col11" class="data row23 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col12" class="data row23 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col13" class="data row23 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col14" class="data row23 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col15" class="data row23 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col16" class="data row23 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col17" class="data row23 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col18" class="data row23 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col19" class="data row23 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col20" class="data row23 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col21" class="data row23 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col22" class="data row23 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col23" class="data row23 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col24" class="data row23 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col25" class="data row23 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col26" class="data row23 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row23_col27" class="data row23 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col0" class="data row24 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col1" class="data row24 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col2" class="data row24 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col3" class="data row24 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col4" class="data row24 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col5" class="data row24 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col6" class="data row24 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col7" class="data row24 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col8" class="data row24 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col9" class="data row24 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col10" class="data row24 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col11" class="data row24 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col12" class="data row24 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col13" class="data row24 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col14" class="data row24 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col15" class="data row24 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col16" class="data row24 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col17" class="data row24 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col18" class="data row24 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col19" class="data row24 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col20" class="data row24 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col21" class="data row24 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col22" class="data row24 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col23" class="data row24 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col24" class="data row24 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col25" class="data row24 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col26" class="data row24 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row24_col27" class="data row24 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col0" class="data row25 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col1" class="data row25 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col2" class="data row25 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col3" class="data row25 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col4" class="data row25 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col5" class="data row25 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col6" class="data row25 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col7" class="data row25 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col8" class="data row25 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col9" class="data row25 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col10" class="data row25 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col11" class="data row25 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col12" class="data row25 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col13" class="data row25 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col14" class="data row25 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col15" class="data row25 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col16" class="data row25 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col17" class="data row25 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col18" class="data row25 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col19" class="data row25 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col20" class="data row25 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col21" class="data row25 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col22" class="data row25 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col23" class="data row25 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col24" class="data row25 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col25" class="data row25 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col26" class="data row25 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row25_col27" class="data row25 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col0" class="data row26 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col1" class="data row26 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col2" class="data row26 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col3" class="data row26 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col4" class="data row26 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col5" class="data row26 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col6" class="data row26 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col7" class="data row26 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col8" class="data row26 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col9" class="data row26 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col10" class="data row26 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col11" class="data row26 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col12" class="data row26 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col13" class="data row26 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col14" class="data row26 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col15" class="data row26 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col16" class="data row26 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col17" class="data row26 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col18" class="data row26 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col19" class="data row26 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col20" class="data row26 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col21" class="data row26 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col22" class="data row26 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col23" class="data row26 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col24" class="data row26 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col25" class="data row26 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col26" class="data row26 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row26_col27" class="data row26 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_08e7b92e_3844_11eb_9c0a_0242ac110002level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col0" class="data row27 col0" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col1" class="data row27 col1" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col2" class="data row27 col2" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col3" class="data row27 col3" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col4" class="data row27 col4" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col5" class="data row27 col5" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col6" class="data row27 col6" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col7" class="data row27 col7" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col8" class="data row27 col8" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col9" class="data row27 col9" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col10" class="data row27 col10" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col11" class="data row27 col11" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col12" class="data row27 col12" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col13" class="data row27 col13" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col14" class="data row27 col14" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col15" class="data row27 col15" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col16" class="data row27 col16" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col17" class="data row27 col17" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col18" class="data row27 col18" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col19" class="data row27 col19" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col20" class="data row27 col20" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col21" class="data row27 col21" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col22" class="data row27 col22" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col23" class="data row27 col23" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col24" class="data row27 col24" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col25" class="data row27 col25" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col26" class="data row27 col26" >0</td>
                        <td id="T_08e7b92e_3844_11eb_9c0a_0242ac110002row27_col27" class="data row27 col27" >0</td>
            </tr>
    </tbody></table>



I really love the pandas heatmap representation of the data to really understand the structure!

Interestingly, it is harder for me to recognize the image as a shoe zoomed in though - a true [Monet](https://en.wikipedia.org/wiki/Claude_Monet).


```python
show_image(trainX[random_ix[0]])
```




    <AxesSubplot:>




    
![png](/post/2020-12-06-image-classification-from-scratch_files/output_23_1.png)
    


Now that we are familiar with our data, let's see if we can train a model to classify these images.

## Approach - Linear Regression

Here we will train a linear model to determine if an image is a sneaker (7) or a shirt (6). 
Linear regressions are usually used to predict continuous outcomes and have the form 

$$\hat{y_{i} = \beta_{0} + \beta_{1} x_{i,1} + ... + \beta_{p} x_{i,p}$$

or, in matrix algebra

\\(\textbf{y} = X \bf{\beta} + \bf{\epsilon}\\)

Since we are not predicting a continuous variable (ie. dollars spent on popcorn, GDP, net worth of celebrities, etc.) but instead estimating class assignment, we can map these predictions to probabilities (i.e. values in [0,1]) with the sigmoid function.

$$S(x) = \frac{1}{1+e^{-x}}$$

If that function looks weird, that's ok! Let's just plot it to make sure it's doing what we think it's doing.


```python
def sigmoid_function(x):
    return(1/(1+np.exp(-x)))
```


```python
x=np.arange(-6,6,0.1)
plt.plot(x,sigmoid_function(x))
```




    [<matplotlib.lines.Line2D at 0x7efec02362b0>]




    
![png](/post/2020-12-06-image-classification-from-scratch_files/output_28_1.png)
    


Yup! Looks like $s(x)$ is bounded by 0 and 1.

## Fitting our Logistic Regression with Stocastic Gradient Descent (SGD)

So we have a model (logistic regression) - great! Now - how do we fit it? - Stocastic Gradient Descent (SGD).

For those who may not be familiar, SGD is an iterative approach to model fitting that is very popular in deep/machine learning. The basic premise of SGD is we make predictions using our model and subsets of our training data (minibatchs), we determine how 'good' our model did by evaluating the loss for each batch, and slowly move model parameters to minimize the loss. 

Specifically, SGD involves the following steps:

a.) initialize (or generate) random weights
 - This would be our beta coefficients above ($\beta_1$ through $\beta_k$ we will refer to as our weights, $\beta_0$ as our bias).
 
b.) for each batch of data, use our model and weights to predict each image's class

c.) compute how the loss (how good or bad our model was)

d.) compute the gradient* for each weight

e.) update our weights using the gradients

f.) repeat steps b-e until we have a good model


#### Step 0 - Let's get our data in the correct formats

In order to run these experiments we will need to transform our training features to be a single vector of size 784 (that's just 28 times 28) instead of the current 28X28 tensor. 
We will also transform our target to 1 when the image is a sneaker and 0 when a shoe.


```python
# for the training set
train_x = trainX.view(-1,28*28).float()
train_y = torch.where(trainy==torch.tensor(6),torch.tensor(1),torch.tensor(0)).unsqueeze(1)
train_x.shape,train_y.shape
```




    (torch.Size([10800, 784]), torch.Size([10800, 1]))




```python
# for the validation set
valid_x = validX.view(-1, 28*28).float()
valid_y = torch.where(validy==torch.tensor(6),torch.tensor(1),torch.tensor(0)).unsqueeze(1)
valid_x.shape,valid_y.shape
```




    (torch.Size([1200, 784]), torch.Size([1200, 1]))



## a.) initialize (or generate) random weights

In order to start SGD we need to have some random parameters initialized.


```python
def initialize_params(size): 
    return (torch.randn(size)).requires_grad_()
```

Now we can initialize our weights and bias


```python
weights = initialize_params((28*28,1))
bias = initialize_params(1)
weights.shape,bias.shape
```




    (torch.Size([784, 1]), torch.Size([1]))



## b.) for each batch of data, use our model and weights to predict each image's class

Next, lets define our model.

We have: 
- our features: $X$ (1x784)
- our weights: $\beta$ (784x1)
- our bias: $\epsilon$ (1x1)

$$\hat{P(x_i = \text{Shirt})} = S(X \bf{\beta} + \bf{\epsilon})$$

where $S(x)$ is the sigmoid function defined as

$S(x) = \frac{1}{1+e^{-z}}$


```python
def logistic_regression(x):
    # x is our pixel values
    # weights are our beta coefficients
    # bias is our bias term 
    return (x@weights + bias).sigmoid()
```

## c.) compute how the loss (how good or bad our model was)

Now, we actually have enough infrastructure (parameters and a model which uses the parameters) to make some (bad) 'predictions' using our `logistic_regression` model.

As my 3rd grade teacher would say, let's make mistakes! (miss frizzle)


```python
model_predictions = logistic_regression(train_x)
model_predictions[0:3]
```




    tensor([[1.],
            [0.],
            [1.]], grad_fn=<SliceBackward>)



Drum roll please!

And our overall accuracy is:


```python
corrects = (model_predictions>0.0) == train_y
corrects.float().mean().item()
```




    0.6226851940155029



So the accuracy here isn't great, but hey what can we expect, we are essentially randomly determining the image's class.

Now that we have the ability to make predictions with our model we need to assess the performance in a very granular way. Accuracy is a metric that tells us how we are performing in a very human interpretable way, but we can actually change the weights and have the same accuracy. Imagine 2 models trying to assess if an image of a Yorkie is a dog or a cat. Model A may say that the image is a dog with 57% confidence while model B may think the image is a dog with 99% confidence. Assuming we have a decision boundary of 50%, both models would be 'correct' but clearly model B deserves more credit. 

A good loss function will capture these differences resulting in different 'loss' for the same 'accuracy'.
The simplest loss function is the mean absolute error (sometimes known as L1 loss). So to keep things simple we will use that.


```python
def l1_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return (targets-predictions).abs().mean()
```

## d.) compute the gradient for each weight

For SGD we will feed our model a series of random slices of data, compute the gradeint for each parameter (weights), updating them accordingly in order to minimize the models loss for each iteration of the entire data set (epoch).

I'm going to use fast.ai's `DataLoader` class here to construct this iterator for both the training and validation sets. 


```python
dataset = list(zip(train_x,train_y))
dl = DataLoader(dataset, batch_size=270,shuffle=True)
valid_dset = list(zip(valid_x,valid_y))
valid_dl = DataLoader(valid_dset, batch_size=240,shuffle=True)
```

We can look at the first iteration of the dataloader which shows us that we have 270 rows of size 784 (28X28)


```python
xb,yb = first(dl)
xb.shape,yb.shape
```




    (torch.Size([270, 784]), torch.Size([270, 1]))



#### Quick aside: How SGD works.
Without getting too much into the weeds here, SGD works by 
 - making predictions using the given weights
 - computing how 'off' those predictions were by calculating a loss
 - computing the gradient (derivative) of the loss
 - moving the weights to minimize the loss 
     - new_weights = old_weights - weights_derivative\*constant

The derivative** is computed automatically for us by Pytorch using the `.backward()` attribute. The inner workings of this 'autograd' out of the scope of this post.
The 'constant' we are referring to here is also known as the 'learning rate' in the literature. 

We can define a function that computes the model parameter gradients (derivative) with respect to our loss function.


```python
def calc_grad(x, y, model_fn, loss_fn=l1_loss):
    preds = model_fn(x)
    loss = loss_fn(preds, y)
    loss.backward()
```

Great! Once we have the gradient we can use them to update the parameters in order to minimize the loss.

## e.) update our weights using the gradients


```python
def update_parameters(parameters,lr):
    for param in parameters:
        param.data -= param.grad*lr
        param.grad.zero_()
```

## f.) repeat steps b-e until we have a good model

Now we can put these pieces together.

For each batch within an epoch, we need to compute the loss of our model, determine the parameters' gradient (`calc_grad`) and update the parameters accordingly (steps b-e).

We can define our `train epoch` function below to do this for us.


```python
def train_epoch(model_fn,lr,params,training_data,valid_data):
    for x,y in training_data:
        calc_grad(x,y,model_fn,loss_fn = l1_loss)
        update_parameters(params,lr)
```

While not technically necessary to generate our model, it would be helpful to print out how well our model is performing as we train it. We can define an `accuracy` function to print the accuracy of our model against some some held out data.


```python
def accuracy(model_fn,x,y):
    preds = (model_fn(x)>0.5)
    acc = (preds == y).float().mean()
    print(f'accuracy: {round(acc.item()*100,2)}%')
```

## Let's run our model!

So we can now use our `initialize_params` function to initialize our parameters (weights, bias), and our `train_epoch` function to fit the model for several epochs.


```python
torch.manual_seed(50)
weights = initialize_params((28*28,1))
bias = initialize_params(1)
```


```python
epochs = 20
lr = 0.1
for i in range(epochs):
    train_epoch(model_fn = logistic_regression,
               lr=lr,
               params = (weights,bias),
               training_data=dl,
               valid_data=valid_dl)
    accuracy(model_fn = logistic_regression,
                  x=valid_x,
                  y=valid_y)
```

    accuracy: 46.92%
    accuracy: 49.83%
    accuracy: 54.67%
    accuracy: 60.92%
    accuracy: 72.0%
    accuracy: 80.25%
    accuracy: 83.83%
    accuracy: 89.33%
    accuracy: 91.83%
    accuracy: 92.67%
    accuracy: 93.33%
    accuracy: 93.67%
    accuracy: 93.83%
    accuracy: 94.5%
    accuracy: 94.5%
    accuracy: 94.42%
    accuracy: 94.75%
    accuracy: 95.25%
    accuracy: 95.25%
    accuracy: 95.42%


Wow! After 20 epochs (that took about 3 seconds to run) we are at over 97% accuracy in our out of sample validation set.

Now, let's see how we perform on our test set.


```python
accuracy(model_fn = logistic_regression,
                  x=train_x,
                  y=train_y)
```

    accuracy: 94.84%


Voilà!  Over 95% - not bad!

We now have an image classifier built from scratch! In building this tool, we also implemented SGD and found our performance to be pretty great for a few lines of code! Hopefully, you can take the information in this post and build your own classifier or extend this one!

There were a few aspects of SGD and image classification that we glanced over, as well as a lot of tricks modern image recognition models use to improve speed and performance. But the core of these modern models is exactly what we just saw!

Other Notes:

\* The gradient here is the slope of the loss with respect to the model parameters. If we can compute the slope of the loss, we can move our parameters where the slope is negative to reduce our loss and yield a better model.

Example: we know loss(x,p) has derivative -2 at current p. Therefore increating the value of p a small amount (\\(p_2=p-(-2) * \epsilon\\) ) should lower the value of loss(x,\\(p_2\\)).

\** We can't have SGD without computing the gradient. The Pytorch [autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) is what computes the gradients (AKA derivatives) for us. We totally took this crucial part for granted in this post, but it's a really important and impressive piece of technology that drives modern deep learning.

