from fastai.vision.all import load_learner
import gradio as gr
# import pathlib

# For the posix path error: when you train your model on colab/gradient and download it, then do inference on Windows.
# Redirect PosixPath to WindowsPath:
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


# Data below sourced from:
# URL = 'https://dog.ceo/api/breeds/list/all'
# To remain consistent, I initialized it as a tuple (previously broke when the API was utilized to get dog breeds)
dog_breeds = tuple(['affenpinscher', 'afghan hound', 'african', 'airedale', 'akita', 'american terrier', 'appenzeller', 'australian cattledog', 'australian terrier', 'basenji', 'basset hound', 'beagle', 'bedlington terrier', 'bernese mountain', 'bichon frise', 'blenheim spaniel', 'blood hound', 'bluetick', 'border collie', 'border terrier', 'borzoi', 'boston bulldog', 'bouvier', 'boxer', 'brabancon', 'briard', 'brittany spaniel', 'bull mastiff', 'cairn terrier', 'cardigan corgi', 'caucasian ovcharka', 'cavapoo', 'chesapeake retriever', 'chihuahua', 'chow', 'clumber', 'cockapoo', 'cocker spaniel', 'coonhound', 'cotondetulear', 'curly retriever', 'dachshund', 'dalmatian', 'dandie terrier', 'dhole', 'dingo', 'doberman', 'english bulldog', 'english hound', 'english mastiff', 'english setter', 'english sheepdog', 'english springer', 'entlebucher', 'eskimo', 'flatcoated retriever', 'fox terrier', 'french bulldog', 'german pointer', 'germanlonghair pointer', 'germanshepherd', 'giant schnauzer', 'golden retriever', 'gordon setter', 'great dane', 'groenendael', 'havanese', 'husky', 'ibizan hound', 'irish setter', 'irish spaniel', 'irish terrier', 'irish wolfhound', 'italian greyhound', 'italian segugio', 'japanese spaniel', 'japanese spitz', 'keeshond', 'kelpie', 'kerryblue terrier', 'komondor', 'kuvasz', 'labradoodle', 'labrador', 'lakeland terrier', 'lapphund finnish', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese', 'medium poodle', 'mexicanhairless', 'miniature pinscher', 'miniature poodle', 'miniature schnauzer', 'mix', 'newfoundland', 'norfolk terrier', 'norwegian buhund', 'norwegian elkhound', 'norwich terrier', 'otterhound', 'papillon', 'patterdale terrier', 'pekinese', 'pembroke', 'pitbull', 'plott hound', 'pomeranian', 'pug', 'puggle', 'pyrenees', 'redbone', 'rhodesian ridgeback', 'rottweiler', 'russell terrier', 'saluki', 'samoyed', 'schipperke', 'scottish deerhound', 'scottish terrier', 'sealyham terrier', 'sharpei', 'shepherd australian', 'shetland sheepdog', 'shiba', 'shihtzu', 'silky terrier', 'spanish waterdog', 'staffordshire bullterrier', 'standard poodle', 'stbernard', 'sussex spaniel', 'swiss mountain', 'tervuren', 'tibetan mastiff', 'tibetan terrier', 'toy poodle', 'toy terrier', 'vizsla', 'walker hound', 'weimaraner', 'welsh spaniel', 'welsh terrier', 'westhighland terrier', 'wheaten terrier', 'whippet', 'yorkshire terrier'])

def classify_image(img):
    _, _, probs = learn.predict(img)
    return dict(zip(dog_breeds, map(float, probs)))

learn = load_learner('dogIdentifierModel.pkl')

image = gr.components.Image(shape=(396, 396))
label = gr.components.Label()
examples = ['golden-retriever.jpg', 'german-shepherd.jpg', 'doberman.jpg', 'husky.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)