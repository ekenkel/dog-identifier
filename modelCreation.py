from fastcore.all import *
from fastdownload import download_url
from fastai.vision.widgets import *
from fastai.vision.all import *
# import gradio as gr
import os, shutil
import time
import requests
import json
import timm
from azure.cognitiveservices.search.imagesearch import ImageSearchClient as api
from msrest.authentication import CognitiveServicesCredentials as auth


def search_images_bing(key, term, min_sz=128, max_images=110):
    params = {'q': term, 'count': max_images, 'min_height': min_sz, 'min_width': min_sz}
    headers = {"Ocp-Apim-Subscription-Key": key}
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    return L(search_results['value'])


URL = 'https://dog.ceo/api/breeds/list/all'

result = requests.get(url = URL).json()
searchText = []
for val in result['message'].items():
    if len(val[1]) > 0:
        for type in val[1]:
            searchText.append(f'{type} {val[0]}')
    else:
        searchText.append(val[0])

path = Path('Dog_Types')

key = os.environ.get('AZURE_SEARCH_KEY', 'INSERT KEY HERE')

if not path.exists():
    path.mkdir()
else:
    folder = 'Dog_Types'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

for o in searchText:
    try:
        dest = (path / o)
        dest.mkdir(exist_ok=True, parents=True)
        results = search_images_bing(key, f'{o} dog photo')
        download_images(dest, urls=results.attrgot('contentUrl'))
    except shutil.SameFileError:
        pass

for breed in searchText:
    failed = verify_images(get_image_files(f'{path}/{breed}'))
    failed.map(Path.unlink)


dataloaders = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
).dataloaders(path)


learn = vision_learner(dataloaders, 'convnext_tiny_in22k', metrics=error_rate)
learn.fine_tune(18)

learn.export('dogIdentifierModel.pkl')