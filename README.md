---
title: Dog Identifier
emoji: 😻
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 3.1.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Dog Identifier

I created a model using fastai and pytorch that can recognize 146 breeds of dogs.

I used the [Dog API](https://dog.ceo/dog-api/documentation/) to get the breeds, then collected 110 images for each using [Bing Search API](https://docs.microsoft.com/en-us/azure/cognitive-services/bing-web-search/). Due to memory restrictions, I could not increase the dataset.


I randomly split the data:
- 80% training set
- 20% validation set

After resizing the images to be 128 x 128 px, I trained using the ResNet Architecture with 18 layers and 25 epochs.

