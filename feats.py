"""
compute features
"""

import cv2
import sqlite3
import msgpack
import msgpack_numpy as m
import numpy as np
from tqdm import tqdm
from lib import models
from common import get_image_data
from concurrent.futures import ThreadPoolExecutor
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input

model = ResNet50(
    weights='imagenet',
    pooling='avg',
    # exclude classification layer
    include_top=False)

session = models.Session()

conn = sqlite3.connect('feats.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS feats (id integer primary key, feats blob)')

def compute_features(img):
    # resize to imagenet size
    img = cv2.resize(img, (224, 224))
    img = image_utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img)


def compute_features_all():
    limit = 100000
    q = session.query(models.Image.key, models.Image.id).limit(limit)
    with ThreadPoolExecutor() as executor:
        for id, img in tqdm(executor.map(lambda p: (p[0], get_image_data(p[1])), q.all())):
            feats = compute_features(img)

            # empty image
            if feats is None:
                continue

            enc = msgpack.packb(feats, default=m.encode)
            c.execute('INSERT INTO feats VALUES (?, ?)', (id, enc))
            conn.commit()


if __name__ == '__main__':
    compute_features_all()
