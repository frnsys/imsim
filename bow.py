import cv2
import numpy as np

voc = np.load('voc.npy')
detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()
flann_params = dict(algorithm=1, trees=5)
matcher = cv2.FlannBasedMatcher(flann_params, {})
extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)
extract_bow.setVocabulary(voc)


def compute_features(img):
    return extract_bow.compute(img, detect.detect(img))[0]
