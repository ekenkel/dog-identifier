from fastai.vision.all import *
import gradio as gr

searchText = ('Dobermann', 'German Shepherd', 'Golden Retriever', 'Poodle')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(searchText, map(float, probs)))

learn = load_learner('dogIdentifierModel.pkl')

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['golden-retriever.jpg', 'german-shepherd.jpg', 'dobermann.jpg', 'poodle.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)