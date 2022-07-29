from fastai.vision.all import *
import gradio as gr
import requests
import json

URL = 'https://dog.ceo/api/breeds/list/all'

result = requests.get(url = URL).json()
searchText = []
for val in result['message'].items():
    if len(val[1]) > 0:
        for type in val[1]:
            searchText.append(f'{type} {val[0]}')
    else:
        searchText.append(val[0])
searchText.sort()

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(searchText, map(float, probs)))

learn = load_learner('dogIdentifierModel.pkl')

image = gr.components.Image(shape=(192, 192))
label = gr.components.Label()
examples = ['golden-retriever.jpg', 'german-shepherd.jpg', 'doberman.jpg', 'husky.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)