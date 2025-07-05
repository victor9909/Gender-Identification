import training.train_gmm as t_gmm
import testing.test_gmm as test_gmm
import training.train_gaussians as tg
import training.train_discriminative as td
import testing.test_gaussians as test_g
import testing.test_discriminative as test_d
import testing.test_svm as test_s
import training.train_svm as ts
from project_lib.graphics import *
from project_lib.models.pipe import Pipe
from project_lib.models.svm_models import RadialKernelBasedSvm
from training.train_calibration import *
from training.train_fusion import *
from testing.test_fusion import make_fusion_eval
from testing.test_calibration import make_calibration_eval

labels = {
    0: "Male",
    1: "Female"
}

colors = {
    0: "Blues",
    1: "Reds"
}

if __name__ == "__main__":

    DTR, LTR, DTE, LTE = load_dataset()

    print("====== Dataset Information ======")
    check_datasets(DTR, LTR, DTE, LTE)

    # Plots of Dataset Analysis Part
    # ======================================================== #
    # Plotting Histograms
    plot_histograms(DTR, LTR, Preprocessing.RAW, labels)
    plot_histograms_dataset(DTR, LTR, Preprocessing.LDA, labels)
    # Plotting Scatters
    plot_scatter(DTR, LTR)
    # Plotting Heatmaps
    plot_heatmaps(DTR, LTR, labels, colors)
    # Plotting Fraction of Explained Variance
    plot_fraction_explained_variance_pca(DTR, LTR)
    # ======================================================== #

    # Classifiers Performances Part
    # ======================================================== #
    # Training all gaussian models on the report
    tg.train(DTR, LTR, DTE, LTE)
    # Training all discriminative models on the report
    td.train(DTR, LTR, DTE, LTE)
    # Training all SVM models on the report
    ts.train(DTR, LTR, DTE, LTE)
    # Training all GMM models on the report
    t_gmm.train(DTR, LTR, DTE, LTE)
    # Calibration of the best models
    make_train_calibration()
    # Fusion of different models
    make_train_fusion()

    # Classifiers Evaluation Part
    # ======================================================== #
    # Testing of all gaussian
    test_g.evaluation(DTR, LTR, DTE, LTE)
    # Testing all discriminative models on the report
    test_d.evaluation(DTR, LTR, DTE, LTE)
    # Testing all SVM models on the report
    test_s.evaluation(DTR, LTR, DTE, LTE)
    # Testing all GMM models on the report
    test_gmm.evaluation(DTR, LTR, DTE, LTE)
    # Fusion of the best models
    make_fusion_eval(LTR, LTE)
    # Calibrate the models
    make_calibration_eval(LTR, LTE)
    # ======================================================== #
