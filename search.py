import sqlite3
import numpy as np
from PIL import Image
from time import time
from tqdm import tqdm
from common import dec_arr
from feats import compute_features
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors

conn = sqlite3.connect('feats.db.bkup')
c = conn.cursor()

print('Preparing index...')
X = []
lookup = []
count = c.execute('SELECT COUNT(id) FROM feats').fetchone()[0]
for id, feats in tqdm(c.execute('SELECT id, feats FROM feats'), total=count):
    feats = dec_arr(feats)[0]
    # not sure why some features are nan
    if np.any(np.isnan(feats)):
        continue
    lookup.append(id)
    X.append(feats)
X = np.vstack(X)

# np.count_nonzero(np.isnan(X))

print('Fitting KNN')
start = time()
knn = NearestNeighbors(n_neighbors=20, n_jobs=8, algorithm='ball_tree')
knn.fit(X)
print('took:', time() - start)

print('saving')
joblib.dump(knn, 'knn.pkl')
# knn = joblib.load('knn.pkl')

if __name__ == '__main__':
    from lib import models
    from common import get_image_data

    fn = 'query.jpg'
    im = Image.open(fn)
    query = compute_features(im).reshape(1, -1)

    neighbors = knn.kneighbors(query, return_distance=True)
    dists = neighbors[0][0]
    idxs = neighbors[1][0]

    session = models.Session()
    for i, (dist, idx) in enumerate(zip(dists, idxs)):
        id = lookup[idx]
        print(id, dist)
        _, key = session.query(models.Image.id, models.Image.key).filter(models.Image.id == id).first()
        img = get_image_data(key)
        img.save('images/{:03d}_{}_{}.png'.format(i, dist, id))