import cv2
import sqlite3
from tqdm import tqdm
from lshash.lshash import LSHash
from bow import compute_features
from common import dec_arr

conn = sqlite3.connect('feats.db')
c = conn.cursor()


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
    results = lsh.query(query, num_results=10, distance_func='centred_euclidean')
    ids_dists = [(r[0][1], r[1]) for r in results]
    import ipdb; ipdb.set_trace()