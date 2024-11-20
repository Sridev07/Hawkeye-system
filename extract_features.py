# Import the functions to calculate feature descriptors
from skimage.feature import hog
# To read file names
import joblib
import argparse as ap
import glob
import os
from config import *
import sys
import numpy as np
sys.modules['sklearn.externals.joblib'] = joblib
import cv2 as cv
# import features directory from data directory
pos_feat_ph = "./data/features/pos"
neg_feat_ph = "./data/features/neg"

if __name__ == "__main__":
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pospath", help="Path to positive images",
            required=True)
    parser.add_argument('-n', "--negpath", help="Path to negative images",
            required=True)
    parser.add_argument('-d', "--descriptor", help="Descriptor to be used -- HOG",
            default="HOG")
    args = vars(parser.parse_args())

    pos_im_path = args["pospath"]
    neg_im_path = args["negpath"]
	
    des_type = args["descriptor"]

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)

    print ("Calculating the descriptors for the positive samples and saving them")
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        # convert to gray scale
        im = cv.imread(im_path)
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        if des_type == "HOG":
            fd = hog(im,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',visualize=False,transform_sqrt=False,feature_vector=True)
        print(fd)
        # print(im_path)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print ("Positive features saved in {}".format(pos_feat_ph))

    print ("Calculating the descriptors for the negative samples and saving them")
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = cv.imread(im_path)
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)      
        if des_type == "HOG":
            fd = hog(im,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',visualize=False,transform_sqrt=False,feature_vector=True)

        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_ph, fd_name)
        print(fd_path)
        joblib.dump(fd, fd_path)
    print ("Negative features saved in {}".format(neg_feat_ph))

    print ("Completed calculating features from training images")
    