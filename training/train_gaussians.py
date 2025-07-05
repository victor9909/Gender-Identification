import os

from project_lib.models.gaussian_models import *
from project_lib.models.pipe import *
from project_lib.preprocessing import *


def train_single_gaussian_model(model, preprocess, DTR, LTR, DTE, LTE):
    pipe_gaussian_model = Pipe(DTR, LTR, DTE, LTE, preprocess, model())
    pipe_gaussian_model.make_train_with_K_fold()


def train(DTR, LTR, DTE, LTE):

    print("Starting training Gaussian Classifiers...")
    dict_preprocess_list = {
        "RAW": [],
        "Z-Scored": [Zscore()],
        "RAW + PCA (12)": [Pca(12)],
        "Z-Scored + PCA(12)": [Zscore(), Pca(12)],
        "RAW + PCA (11)": [Pca(11)],
        "Z-Scored + PCA(11)": [Zscore(), Pca(11)],
        "RAW + PCA (10)": [Pca(10)],
        "Z-Scored + PCA(10)": [Zscore(), Pca(10)],
    }

    for named_preprocess in dict_preprocess_list.keys():
        for gaussian_model in [MVG, NBG, TMVG, TNB]:
            print("========================================================")
            print("Training " + str(gaussian_model()).replace("_", "") + " with " + named_preprocess + " features...")
            train_single_gaussian_model(gaussian_model, dict_preprocess_list[named_preprocess], DTR, LTR, DTE, LTE)
            print("End training " + str(gaussian_model()).replace("_", "") + " with " + named_preprocess + " features")
            print("========================================================")

    print("Finish training on gaussian classifiers...")
