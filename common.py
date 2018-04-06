import cv2
import boto3
import msgpack
import numpy as np
import msgpack_numpy as m

s3 = boto3.resource('s3')
bucket_name = 'vizlab-images'


def get_image_data(key):
    """download image from S3 and return its data"""
    obj = s3.Object(bucket_name, key)
    imdata = obj.get()['Body'].read()
    imdata = np.asarray(bytearray(imdata), dtype=np.uint8)
    return cv2.imdecode(imdata, 0)


def enc_arr(arr):
    return msgpack.packb(arr, default=m.encode)


def dec_arr(enc):
    return msgpack.unpackb(enc, object_hook=m.decode)
