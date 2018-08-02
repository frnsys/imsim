import cv2
import sqlite3
import numpy as np
from tqdm import tqdm
from lshash.lshash import LSHash
from feats import compute_features
from common import dec_arr

conn = sqlite3.connect('feats.db')
c = conn.cursor()

# function to compute chi square dist
def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))
    return d

if __name__ == '__main__':
    n_dim = 40
    lsh = LSHash(40, n_dim)

    print('Preparing index...')
    count = c.execute('SELECT COUNT(id) FROM feats').fetchone()[0]
    for id, feats in tqdm(c.execute('SELECT id, feats FROM feats'), total=count):
        # feats = dec_arr(feats)
        feats = dec_arr(feats)[0]
        lsh.index(feats, extra_data=id)

    print('Searching...')
    fn = 'query.jpg'
    im = cv2.imread(fn, 0)
    query = compute_features(im)

    # distances: euclidean, true_euclidean, cosine, centred_euclidean, l1norm
    # <https://github.com/kayzhu/LSHash/blob/master/lshash/lshash.py#L270>
    results = lsh.query(query, num_results=10, distance_func='centred_euclidean')
    ids_dists = [(r[0][1], r[1]) for r in results]
    import ipdb; ipdb.set_trace()