from fastcore.all import *
from fastdownload import download_url
from fastai.vision.widgets import *
from fastai.vision.all import *
# import gradio as gr
import os, shutil
import time

def search_images(term, max_images=200):
    url = 'https://duckduckgo.com/'
    res = urlread(url,data={'q':term})
    searchObj = re.search(r'vqd=([\d-]+)\&', res)
    requestUrl = url + 'i.js'
    params = dict(l='us-en', o='json', q=term, vqd=searchObj.group(1), f=',,,', p='1', v7exp='a')
    urls,data = set(),{'next':1}
    while len(urls)<max_images and 'next' in data:
        data = urljson(requestUrl,data=params)
        urls.update(L(data['results']).itemgot('image'))
        requestUrl = url + data['next']
        time.sleep(0.2)
    return L(urls)[:max_images]


searchText = ('Golden Retriever', 'German Shepherd', 'Dobermann', 'Poodle')
path = Path('Dog_Types')
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
    dest = (path / o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    resize_images(path / o, max_size=400, dest=path / o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)

dataloaders = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
).dataloaders(path)

dataloaders.show_batch(max_n=6)

learn = vision_learner(dataloaders, resnet18, metrics=error_rate)
learn.fine_tune(20)

learn.export('dogIdentifierModel.pkl')