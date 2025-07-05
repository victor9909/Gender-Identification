
from project_lib.graphics import plot_bayes_error
from project_lib.models.pipe import make_calibration
from project_lib.utils import load_dataset
from project_lib.models.Model import Model
import numpy as np


def make_train_calibration():

    dtr, ltr, dte, lte = load_dataset()
    scores_rbsvm = np.load("scores_train/RadialBasedSVM/ZSCORE/RadialBasedSVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_5_g_0.1.npy")
    scores_svm = np.load("scores_train/SVM/ZSCORE/SVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_10.0.npy")
    scores_lr = np.load("scores_train/LR/ZSCORE/LR_ZSCORE_C0_0.1_C1_0.9_l_1e-05.npy")
    scores_gmm = np.load("scores_train/GMM/ZSCORE/GMM_ZSCORE_components_4_alpha_0.1_psi_0.01.npy")

    plot_bayes_error(scores_rbsvm, Model.RadialBasedSVM, "priors: [0.1, 0.9] - Z-Score")
    plot_bayes_error(scores_svm, Model.SVM, "priors: [0.1, 0.9] - Z-Score")
    plot_bayes_error(scores_lr, Model.LR, "priors: [0.1, 0.9] - Z-Score")
    plot_bayes_error(scores_gmm, Model.GMM, "4 components - Z-Score")

    for i in [[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]]:
        scores_1, labels_1 = make_calibration(scores_rbsvm, ltr, i, "RadialBasedSVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_5_g_0.1")
        plot_bayes_error(scores_1, Model.RadialBasedSVM, "Z-Score calibrated", labels=labels_1)
        scores_2, labels_2 = make_calibration(scores_svm, ltr, i, "SVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_10.0.npy")
        plot_bayes_error(scores_2, Model.SVM, "Z-Score calibrated", labels=labels_2)
        scores_3, labels_3 = make_calibration(scores_lr, ltr, i, "LR_ZSCORE_C0_0.1_C1_0.9_l_1e-05.npy")
        plot_bayes_error(scores_3, Model.LR, "Z-Score calibrated", labels=labels_3)
        scores_4, labels_4 = make_calibration(scores_gmm, ltr, i, "GMM_ZSCORE_components_4_alpha_0.1_psi_0.01.npy")
        plot_bayes_error(scores_4, Model.GMM, "4 components - Z-Score calibrated", labels=labels_4)
