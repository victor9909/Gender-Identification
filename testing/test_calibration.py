import os

import numpy as np
from project_lib.models.pipe import make_eval_calibration
from project_lib.graphics import plot_bayes_error_more_models
from project_lib.models.Model import Model
from project_lib.utils import compute_NDCF, compute_minimum_NDCF


def make_calibration_eval(LTR, LTE):
    scores_test_lr = np.load("scores_test/LR/ZSCORE/LR_ZSCORE_C0_0.1_C1_0.9_l_1e-05.npy")
    scores_test_svm = np.load("scores_test/SVM/ZSCORE/SVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_10.0.npy")
    scores_rbsvm_test = np.load(
        "scores_test/RadialBasedSVM/ZSCORE/RadialBasedSVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_5_g_0.1.npy")

    scores_train_lr = np.load("scores_train/LR/ZSCORE/LR_ZSCORE_C0_0.1_C1_0.9_l_1e-05.npy")
    scores_train_svm = np.load("scores_train/SVM/ZSCORE/SVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_10.0.npy")
    scores_rbsvm_train = np.load(
        "scores_train/RadialBasedSVM/ZSCORE/RadialBasedSVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_5_g_0.1.npy")

    scores_lr_cal = make_eval_calibration(scores_train_lr, LTR, scores_test_lr, LTE, [0.5, 0.5], "LR_ZSCORE_C0_0.1_C1_0.9_l_1e-05")
    scores_svm_cal = make_eval_calibration(scores_train_svm, LTR, scores_test_svm, LTE, [0.5, 0.5], "SVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_10.0")
    scores_rbsvm_cal = make_eval_calibration(scores_rbsvm_train, LTR, scores_rbsvm_test, LTE, [0.5, 0.5], "RadialBasedSVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_5_g_0.1")

    plot_bayes_error_more_models([scores_lr_cal, scores_svm_cal, scores_rbsvm_cal],
                                 [Model.LR.name, Model.SVM.name, Model.RadialBasedSVM.name],
                                 ["r", "b", "g"],
                                 "Calibrated Scores",
                                 labels=LTE,
                                 is_train=False)

    print(round(compute_NDCF(scores_lr_cal, LTE, 0.5, 1, 1), 3))
    print(round(compute_NDCF(scores_lr_cal, LTE, 0.1, 1, 1), 3))
    print(round(compute_NDCF(scores_lr_cal, LTE, 0.9, 1, 1), 3))

    print(round(compute_minimum_NDCF(scores_lr_cal, LTE, 0.5, 1, 1)[0], 3))
    print(round(compute_minimum_NDCF(scores_lr_cal, LTE, 0.1, 1, 1)[0], 3))
    print(round(compute_minimum_NDCF(scores_lr_cal, LTE, 0.9, 1, 1)[0], 3))

    print(round(compute_NDCF(scores_svm_cal, LTE, 0.5, 1, 1), 3))
    print(round(compute_NDCF(scores_svm_cal, LTE, 0.1, 1, 1), 3))
    print(round(compute_NDCF(scores_svm_cal, LTE, 0.9, 1, 1), 3))

    print(round(compute_minimum_NDCF(scores_svm_cal, LTE, 0.5, 1, 1)[0], 3))
    print(round(compute_minimum_NDCF(scores_svm_cal, LTE, 0.1, 1, 1)[0], 3))
    print(round(compute_minimum_NDCF(scores_svm_cal, LTE, 0.9, 1, 1)[0], 3))

    print(round(compute_NDCF(scores_rbsvm_cal, LTE, 0.5, 1, 1), 3))
    print(round(compute_NDCF(scores_rbsvm_cal, LTE, 0.1, 1, 1), 3))
    print(round(compute_NDCF(scores_rbsvm_cal, LTE, 0.9, 1, 1), 3))

    print(round(compute_minimum_NDCF(scores_rbsvm_cal, LTE, 0.5, 1, 1)[0], 3))
    print(round(compute_minimum_NDCF(scores_rbsvm_cal, LTE, 0.1, 1, 1)[0], 3))
    print(round(compute_minimum_NDCF(scores_rbsvm_cal, LTE, 0.9, 1, 1)[0], 3))

