import boto3
from PIL import Image
from lib import models
from flask import Flask, request, render_template
from search import prepare_index, search

s3 = boto3.client('s3')
bucket_name = 'vizlab-images'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

session = models.Session()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
print('Building index...')
knn, lookup = prepare_index(limit=1200000)
print('Done building index')


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        f = request.files['image']
        if f and f.filename and allowed_file(f.filename):
            img = Image.open(f.stream)
            dists, idxs = search(knn, img)

            ids_to_dists = {}
            ids = [lookup[idx] for idx in idxs]
            for dist, id in zip(dists, ids):
                ids_to_dists[id] = dist

            images = session.query(models.Image).filter(models.Image.id.in_(ids)).all()
            images = [
                (img, ids_to_dists[img.id], s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': img.key}, ExpiresIn=1800))
                for img in images]
            return render_template('results.html', images=images)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)