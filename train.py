"""
train a BoW vocab from SIFT features
"""

import cv2
import sqlite3
import numpy as np
from tqdm import tqdm
from common import dec_arr


conn = sqlite3.connect('sift.db')
c = conn.cursor()


def train_vocab(n_dim):
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(n_dim)

    count = c.execute('SELECT COUNT(id) FROM sift').fetchone()[0]
    limit = count

    print('Decoding features...')
    for id, sift in tqdm(c.execute('SELECT id, sift FROM sift WHERE id IN (SELECT id FROM sift ORDER BY RANDOM() LIMIT ?)', [limit]), total=limit):
        sift = dec_arr(sift)
        bow_kmeans_trainer.add(sift.astype(np.float32))

    print('Clustering...')
    voc = bow_kmeans_trainer.cluster()
    np.save('voc.npy', voc)
    print('Saved vocab to voc.npy')


if __name__ == '__main__':
    n_dim = 128
    train_vocab(n_dim)
