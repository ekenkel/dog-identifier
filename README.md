---
title: Dog Identifier
emoji: 🦮
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 3.1.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Dog Identifier



[Open Demo](https://huggingface.co/spaces/ekenkel/dog-identifier)

---

I created a model using fastai and pytorch that can recognize 146 breeds of dogs with 75% accuracy.

I used the [Dog API](https://dog.ceo/dog-api/documentation/) to get the breeds, then collected 105 images for each using [Bing Search API](https://docs.microsoft.com/en-us/azure/cognitive-services/bing-web-search/). Due to memory restrictions, I could not increase the dataset.


I randomly split the data:
- 80% training set
- 20% validation set

I trained using the ConvNeXt-22k convolution neural network and 11 epochs.

I previously attained 75% accuracy. I made improvements by resizing each image to the same dimension (460x460 px) and then doing augmentations on the batches (distortion, lighting, cropping). I also froze the pretrained model (ConvNeXt-22k) layers, and trained my newly added layers for 3 epochs. I saved time and GPU memory by using half-precision floating point where applicable during training.

Before that, my base model used the ResNet architecture with 18 layers and 25 epochs, but was only able to get 63% accuracy.

