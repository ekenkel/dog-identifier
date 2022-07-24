from fastai.vision.all import *
import gradio as gr

searchText = ('Golden Retreiver','German Shepherd', 'Dobermann', 'Golden Doodle')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(searchText, map(float,probs)))

learn = load_learner('dogIdentifierModel.pkl')

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['german-shepherd.jpg', 'golden-retriever.jpg', 'dobermann.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)