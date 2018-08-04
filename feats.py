"""
compute features
"""

import sqlite3
import msgpack
import msgpack_numpy as m
import numpy as np
from PIL import Image
from tqdm import tqdm
from lib import models
from common import get_image_data
from concurrent.futures import ThreadPoolExecutor
from keras.applications.resnet50 import ResNet50, preprocess_input

print('Loading model...')
im_shape = (244, 244)
model = ResNet50(
    input_shape=(244,244,3),
    pooling='avg',
    weights='imagenet',
    # exclude classification layer
    include_top=False)
print('Done loading model')

session = models.Session()

conn = sqlite3.connect('feats.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS feats (id integer primary key, feats blob)')
processed = set(c.execute('SELECT id FROM feats').fetchall())

def compute_features(img):
    img = img.convert('RGB')
    img = img.resize(im_shape, Image.ANTIALIAS)
    img = np.array(img)
    img = np.expand_dims(img, axis=0).astype(np.float)
    img = preprocess_input(img)
    feats = model.predict(img)
    return feats[0].flatten()


def process(entry):
    id, key = entry
    ok = id in processed
    img = None if ok else get_image_data(key)
    return id, img, ok


def compute_features_all():
    limit = 5000
    q = session.query(models.Image.id, models.Image.key).limit(limit)
    with ThreadPoolExecutor() as executor:
        for id, img, ok in tqdm(executor.map(process, q.all()), total=q.count()):
            if ok: continue
            feats = compute_features(img)

            # empty image
            if feats is None: continue

            enc = msgpack.packb(feats, default=m.encode)
            c.execute('INSERT INTO feats VALUES (?, ?)', (id, enc))
            conn.commit()


if __name__ == '__main__':
    compute_features_all()
