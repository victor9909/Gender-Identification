import os

import numpy as np
from project_lib.models.pipe import make_eval_fusion
from project_lib.graphics import plot_bayes_error, plot_det, plot_bayes_error_more_models
from project_lib.models.Model import Model
from project_lib.utils import compute_NDCF


def make_fusion_eval(LTR, LTE):

    scores_test_gmm = np.load("scores_test/GMM/ZSCORE/GMM_ZSCORE_components_4_alpha_0.1_psi_0.01.npy")
    scores_test_lr = np.load("scores_test/LR/ZSCORE/LR_ZSCORE_C0_0.1_C1_0.9_l_1e-05.npy")
    scores_test_svm = np.load("scores_test/SVM/ZSCORE/SVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_10.0.npy")

    scores_gmm_train = np.load("scores_train/GMM/ZSCORE/GMM_ZSCORE_components_4_alpha_0.1_psi_0.01.npy")
    scores_lr_train = np.load("scores_train/LR/ZSCORE/LR_ZSCORE_C0_0.1_C1_0.9_l_1e-05.npy")
    scores_svm_train = np.load("scores_train/SVM/ZSCORE/SVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_10.0.npy")

    scores_gmm_lr = make_eval_fusion([scores_gmm_train, scores_lr_train], LTR, [scores_test_gmm, scores_test_lr], LTE,
                     [0.5, 0.5], "Fusion GMM+LR")
    scores_lr_svm = make_eval_fusion([scores_lr_train, scores_svm_train], LTR, [scores_test_lr, scores_test_svm], LTE,
                     [0.5, 0.5], "Fusion LR+SVM")
    scores_gmm_svm = make_eval_fusion([scores_gmm_train, scores_svm_train], LTR, [scores_test_gmm, scores_test_svm], LTE,
                     [0.5, 0.5], "Fusion GMM+SVM")
    scores_gmm_svm_lr = make_eval_fusion([scores_gmm_train, scores_svm_train, scores_lr_train], LTR,
                                      [scores_test_gmm, scores_test_svm, scores_test_lr], LTE,
                                      [0.5, 0.5], "Fusion GMM+SVM+LR")

    plot_bayes_error(scores_gmm_lr, Model.Fusion, "Fusion GMM+LR", labels=LTE, is_train=False)
    plot_bayes_error(scores_lr_svm, Model.Fusion, "Fusion SVM+LR", labels=LTE, is_train=False)
    plot_bayes_error(scores_gmm_svm, Model.Fusion, "Fusion GMM+SVM", labels=LTE, is_train=False)
    plot_bayes_error(scores_gmm_svm_lr, Model.Fusion, "Fusion SVM+LR+GMM", labels=LTE, is_train=False)

    # Plotting DET
    scores_gmm_test = np.load("scores_test/GMM/ZSCORE/GMM_ZSCORE_components_4_alpha_0.1_psi_0.01.npy")
    scores_tmvg_test = np.load("scores_test/TMVG/ZSCORE/TMVG_ZSCORE.npy")
    scores_rbsvm_test = np.load("scores_test/RadialBasedSVM/ZSCORE/RadialBasedSVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_5_g_0.1.npy")

    plot_det([scores_gmm_test, scores_rbsvm_test, scores_tmvg_test], LTE, ["GMM (4 components; ZScore)", "TMVG (ZScore)", "RBSVM (C=5; prior_true=0.9; ZScore)"], ["r", "b",  "g"], "det plots for RBSVM, GMM and TMVG")

    scores_test_lr = np.load("scores_test/LR/ZSCORE/LR_ZSCORE_C0_0.1_C1_0.9_l_1e-05.npy")
    scores_test_svm = np.load("scores_test/SVM/ZSCORE/SVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_10.0.npy")
    scores_rbsvm_test = np.load("scores_test/RadialBasedSVM/ZSCORE/RadialBasedSVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_5_g_0.1.npy")

    plot_bayes_error_more_models(
        [scores_rbsvm_test, scores_test_svm, scores_test_lr],
        [Model.RadialBasedSVM.name, Model.SVM.name, Model.LR.name],
        ["r", "b", "g"],
        "Uncalibrated Scores",
        labels=LTE,
        is_train=False
    )

    for file_name in os.listdir("scores_test/fusion"):
        print(file_name)
        scores = np.load("scores_test/fusion/" + file_name)
        print(round(compute_NDCF(scores, LTE, 0.5, 1, 1), 3))
        print(round(compute_NDCF(scores, LTE, 0.1, 1, 1), 3))
        print(round(compute_NDCF(scores, LTE, 0.9, 1, 1), 3))


