"""
computes SIFT features for all images in the database.
"""

import cv2
import sqlite3
import numpy as np
from tqdm import tqdm
from lib import models
from sqlalchemy.sql.expression import func
from concurrent.futures import ThreadPoolExecutor
from common import get_image_data, enc_arr


session = models.Session()

# create db
conn = sqlite3.connect('sift.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS sift (id integer primary key, sift blob)')




def compute_sift():
    detect = cv2.xfeatures2d.SIFT_create()
    extract = cv2.xfeatures2d.SIFT_create()

    # alternatively:
    # detect = cv2.xfeatures2d.SURF_create()
    # extract = cv2.xfeatures2d.SURF_create()

    q = session.query(models.Image.id, models.Image.key)

    # sort randomly, for selecting a random sample
    q = q.order_by(func.random())

    with ThreadPoolExecutor() as executor:
        for id, img in tqdm(executor.map(lambda p: (p[0], get_image_data(p[1])), q.all())):
            sift = extract.compute(img, detect.detect(img))[1]

            # empty image
            if sift is None:
                continue

            # reduces size a bit
            # (by default dtype is float32)
            # while, in my experiments, keeping the contents the same
            # i.e. they are really only using integers
            # this is only true for SIFT! SURF requires floats.
            assert np.all(sift == sift.astype(np.uint8))
            sift = sift.astype(np.uint8)

            enc = enc_arr(sift)
            c.execute('INSERT INTO sift VALUES (?, ?)', (id, enc))
            conn.commit()


if __name__ == '__main__':
    compute_sift()
