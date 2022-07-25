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

I created a model using fastai and pytorch that can recognize 4 different types of dogs:
- Golden Retriever
- German Shepherd
- Dobermann
- Poodle



Based on 200 images of each, I randomly split the data:
- 80% training set
- 20% validation set

After resizing the images to be 128 x 128 px, trained using the ResNet Architecture with 18 layers.

