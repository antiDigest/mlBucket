# import the necessary packages
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse

import cv2
from glob import glob
import random


def clusterImage(k, filename, image, pixels):
    (h, w) = image.shape[:2]
    print("Running with k = " + str(k))

    clt = KMeans(n_clusters=k)
    clt.fit(pixels)

    labels = clt.predict(pixels)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant = quant.reshape((h, w, 3))
    # quant = cv2.resize(quant, (w / 3, h / 3))

    cv2.imwrite("clusteredImages/" + filename + "_k_" + str(k) +
                ".jpg", quant)
    # cv2.waitKey(0)
    return


def segmentImage(path, k):
    filename = path.split('/')[1].split('.')[0]

    print("Image: " + filename)
    image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    (h, w) = image.shape[:2]

    pixels = image.reshape((h * w, 3))

    image = image.reshape((h, w, 3))
    # image = cv2.resize(image, (w / 3, h / 3))

    if(k == 'auto'):
        k = 1
        for i in range(4):
            # if k < 5:
            k = random.randint(k + 1, k + 5)
            # elif k < 10:
            #     k = random.randint(k + 1, k + 3)
            # else:
            #     k = random.randint(k + 1, k + 2)
            clusterImage(int(k), filename, image, pixels)
    else:
        clusterImage(int(k), filename, image, pixels)

    return


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, default='auto',
                    help="Path to the image\nDefault: 'auto' -- runs on all images in 'images/' folder")
    ap.add_argument("-c", "--clusters", required=False, type=str, default='auto',
                    help="# of clusters\nDefault: 'auto' -- tests on a couple of random values of k")
    args = vars(ap.parse_args())

    if(args['image'] == 'auto'):
        for path in glob("images/*"):
            segmentImage(path, args['clusters'])
    else:
        segmentImage(args['image'], args['clusters'])
