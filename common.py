import boto3
import msgpack
import msgpack_numpy as m
from PIL import Image

s3 = boto3.resource('s3')
bucket_name = 'vizlab-images'


def get_image_data(key):
    """download image from S3 and return its data"""
    try:
        obj = s3.Object(bucket_name, str(key))
        imdata = obj.get()['Body']
        return Image.open(imdata)
    except s3.meta.client.exceptions.NoSuchKey:
        print('No S3 object found for key:', key)
        return


def enc_arr(arr):
    return msgpack.packb(arr, default=m.encode)


def dec_arr(enc):
    return msgpack.unpackb(enc, object_hook=m.decode)
