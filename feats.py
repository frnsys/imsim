"""
compute BoW features for images
"""

import sqlite3
import msgpack
import msgpack_numpy as m
from tqdm import tqdm
from lib import models
from bow import compute_features
from common import get_image_data
from concurrent.futures import ThreadPoolExecutor


session = models.Session()


conn = sqlite3.connect('feats.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS feats (id integer primary key, feats blob)')




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
