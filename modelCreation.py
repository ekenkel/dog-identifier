from fastcore.all import *
from fastdownload import download_url
from fastai.vision.widgets import *
from fastai.vision.all import *
import os
import requests

def search_and_download_images(api_key, breeds, download_path, num_images=10):
    headers = {
        'Ocp-Apim-Subscription-Key': api_key
        }
    
    for breed in breeds:
        # Create a directory for the breed if it doesn't exist
        breed_dir = os.path.join(download_path, breed)
        os.makedirs(breed_dir, exist_ok=True)
        
        # Initialize a counter for the number of successfully downloaded images
        downloaded_count = 0
        
        # Make the API request
        params = {
            'q': f'{breed} dog',
            'count': num_images
        }
        response = requests.get('https://api.bing.microsoft.com/v7.0/images/search', headers=headers, params=params)
        
        # L() is from fastai 
        results = L(response.json()['value'])
        download_images(breed_dir, urls=results.attrgot('contentUrl'))


URL = 'https://dog.ceo/api/breeds/list/all'

# Get the breeds of all the dogs (some of them require reformatting)
result = requests.get(url = URL).json()
dog_breeds = []
for val in result['message'].items():
    if len(val[1]) > 0:
        for type in val[1]:
            dog_breeds.append(f'{type} {val[0]}')
    else:
        dog_breeds.append(val[0])

path = Path('Dog_Types')

api_key = os.environ.get('AZURE_SEARCH_KEY', 'INSERT KEY HERE')

search_and_download_images(api_key, dog_breeds, path, num_images=200)

# Ensure that all images are able to be opened. 
# If they cannot be opened, remove them
for breed in dog_breeds:
    failed = verify_images(get_image_files(f'{path}/{breed}'))
    failed.map(Path.unlink)


# Load the data into fastai datablock
# In this we randomly split the data into train, validation, and test
# Resize the data to be 396x396 px
# Also perform tansforms (in this way, we are able to get imperfect data to train on)
dataloaders = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(396),
    batch_tfms=aug_transforms(size=396, min_scale=0.75)
).dataloaders(path)

# Load the data into the Convnext-22k
# NOTE: When this happens, fastai will adjust the start of the model according to the input size of your data
learn = vision_learner(dataloaders, 'convnext_tiny_in22k', metrics=error_rate).to_fp16()
# Because of this, for 3 epochs, I decided to freeze the weights of the pretrained model (as this has already been optimized)
# Only the additional layers that were added will adjust in the first 3 epochs
# After the 3 epochs, the model will update all weights
learn.fine_tune(8, freeze_epochs=3)

learn.export('dogIdentifierModel.pkl')