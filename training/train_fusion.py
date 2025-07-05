from project_lib.graphics import plot_bayes_error
from project_lib.models.pipe import make_fusion, make_calibration
from project_lib.utils import load_dataset
from project_lib.models.Model import Model
import numpy as np


def make_train_fusion():

    dtr, ltr, dte, lte = load_dataset()
    np.random.seed(27)
    idx = np.random.permutation(ltr.size)

    scores_svm = np.load("scores_train/SVM/ZSCORE/SVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_10.0.npy")
    scores_lr = np.load("scores_train/LR/ZSCORE/LR_ZSCORE_C0_0.1_C1_0.9_l_1e-05.npy")
    scores_gmm = np.load("scores_train/GMM/ZSCORE/GMM_ZSCORE_components_4_alpha_0.1_psi_0.01.npy")

    scores_1, labels_1 = make_fusion([scores_gmm, scores_svm, scores_lr], ltr, [0.5, 0.5], "GMM_SVM_LR")
    plot_bayes_error(scores_1, Model.Fusion, "Fused GMM SVM LR", labels_1)
    scores_1_calibrated, labels_1_calibrated = make_calibration(scores_1, ltr[idx], [0.5, 0.5], "Fusion_GMM_SVM_LR")
    plot_bayes_error(scores_1_calibrated, Model.Fusion, "Fused GMM SVM LR Calibrated", labels=labels_1_calibrated)

    scores_2, labels_2 = make_fusion([scores_gmm, scores_lr], ltr, [0.5, 0.5], "GMM_LR")
    plot_bayes_error(scores_2, Model.Fusion, "Fused GMM LR", labels_2)
    scores_2_calibrated, labels_2_calibrated = make_calibration(scores_2, ltr[idx], [0.5, 0.5], "Fusion_GMM_LR")
    plot_bayes_error(scores_2_calibrated, Model.Fusion, "Fused GMM LR Calibrated", labels_2_calibrated)

    scores_3, labels_3 = make_fusion([scores_gmm, scores_svm], ltr, [0.5, 0.5], "GMM_SVM")
    plot_bayes_error(scores_3, Model.Fusion, "Fused GMM SVM", labels_3)
    scores_3_calibrated, labels_3_calibrated = make_calibration(scores_3, ltr[idx], [0.5, 0.5], "Fusion_GMM_SVM")
    plot_bayes_error(scores_3_calibrated, Model.Fusion, "Fused GMM SVM LR Calibrated", labels_3_calibrated)

    scores_4, labels_4 = make_fusion([scores_lr, scores_svm], ltr, [0.5, 0.5], "LR_SVM")
    plot_bayes_error(scores_4, Model.Fusion, "Fused SVM LR", labels_4)
    scores_4_calibrated, labels_4_calibrated = make_calibration(scores_4, ltr[idx], [0.5, 0.5], "Fusion_LR_SVM")
    plot_bayes_error(scores_4_calibrated, Model.Fusion, "Fused SVM LR Calibrated", labels_4_calibrated)
