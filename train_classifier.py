# Import the required modules
# from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import joblib
import argparse as ap
import glob
import os
from config import *
import numpy as np
import sys
# import NDArrayWrapper
sys.modules['sklearn.externals.joblib'] = joblib
# from sklearn.utils.fixes import NDArrayWrapper
# sys.modules['sklearn.externals.joblib.numpy_pickle_compat'] = NDArrayWrapper
model_path = "C:\\Users\\Parth\\Desktop\\changedcrick\\cricket-umpire-assistance\\data\\models\\newsvm.model"
if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--posfeat", help="Path to the positive features directory", required=True)
    parser.add_argument('-n', "--negfeat", help="Path to the negative features directory", required=True)
    parser.add_argument('-c', "--classifier", help="Classifier to be used", default="LIN_SVM")
    args = vars(parser.parse_args())

    pos_feat_path =  args["posfeat"]
    neg_feat_path = args["negfeat"]

    # Classifiers supported
    clf_type = args['classifier']
    print(clf_type)

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        feat_path = feat_path.replace("\\", "/")
        # print(feat_path)
        fd = joblib.load(feat_path)
        # print(type(fd),"fd.type")
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
        feat_path = feat_path.replace("\\", "/") 
        fd = joblib.load(feat_path)
        print(type(fd),"fd")
        # get the value of fd
        # print(fd,"fd")
        # print(type(fd),"fd.type")
        fds.append(fd)
        labels.append(0)
        
    if clf_type == "LIN_SVM":
        clf = LinearSVC()
        print ("Training a Linear SVM Classifier")
        fds = np.array(fds)
        labels = np.array(labels)
        # print(fds.shape,"fds")
        # print(labels.shape,"labels")
        i=0
        for sublist in fds:
            print(len(sublist))
            i+=1
            print(i,"i check")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
            print(os.path.split(model_path)[0])
        print(os.path.split(model_path)[0])
        # print(clf)
        joblib.dump(clf, model_path)
        print ("Classifier saved to {}".format(model_path))
