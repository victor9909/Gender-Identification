import matplotlib.pyplot as plt

from project_lib.models.discriminative_models import *
from project_lib.models.pipe import *
from project_lib.graphics import compute_std_minCDF_plot_for_model
from project_lib.validation_utils import compute_minCDF_for_discriminative_models


def train_single_logistic_regression_model(priors, preprocess, DTR, LTR, DTE, LTE, l_val):
    pipe_lr = Pipe(DTR, LTR, DTE, LTE, preprocess, LR(priors, l_val))
    pipe_lr.make_train_with_K_fold()


def train_single_quadratic_logistic_regression_model(priors, preprocess, DTR, LTR, DTE, LTE, l_val):
    pipe_qlr = Pipe(DTR, LTR, DTE, LTE, preprocess, QLR(priors, l_val))
    pipe_qlr.make_train_with_K_fold()


def plotting_train_for_logistic_regression():
    for priors in [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]:
        compute_std_minCDF_plot_for_model(Model.QLR, ["raw"], {"prior_t": priors[1]})
        compute_std_minCDF_plot_for_model(Model.QLR, ["pca_m_12"], {"prior_t": priors[1]})
        compute_std_minCDF_plot_for_model(Model.QLR, ["ZScore"], {"prior_t": priors[1]})
        compute_std_minCDF_plot_for_model(Model.QLR, ["ZScore", "pca_m_12"], {"prior_t": priors[1]})


def plotting_train_for_quadratic_logistic_regression():
    for priors in [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]:
        compute_std_minCDF_plot_for_model(Model.LR, ["raw"], {"prior_t": priors[1]})
        compute_std_minCDF_plot_for_model(Model.LR, ["pca_m_12"], {"prior_t": priors[1]})
        compute_std_minCDF_plot_for_model(Model.LR, ["ZScore"], {"prior_t": priors[1]})
        compute_std_minCDF_plot_for_model(Model.LR, ["ZScore", "pca_m_12"], {"prior_t": priors[1]})


def train(DTR, LTR, DTE, LTE):
    print("Training discriminative models...")
    l_val = np.logspace(-5, 5, num=51)
    dict_preprocess_list = {
        "RAW": [],
        "Z-Scored": [Zscore()],
        "RAW + PCA (12)": [Pca(12)],
        "Z-Scored + PCA(12)": [Zscore(), Pca(12)],
    }

    # Logistic Regression Training
    print("================== Start LR Training ===================")
    for named_preprocess in dict_preprocess_list:
        for x in l_val:
            for priors in [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]:
                print("========================================================")
                print("Training " + str(LR([], None)).replace("_", "") +
                      "(πT=" + str(priors[0]) + "; λ=" + str(x) + ") with " + named_preprocess + " features...")
                train_single_logistic_regression_model(priors, dict_preprocess_list[named_preprocess], DTR, LTR, DTE,
                                                       LTE, x)
                print("========================================================")
    print("=================== End LR Training ====================")

    # Quadratic Logistic Regression Training
    print("================== Start QLR Training ==================")
    for named_preprocess in dict_preprocess_list:
        for x in l_val:
            for priors in [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]:
                print("========================================================")
                print("Training " + str(QLR([], None)).replace("_", "") +
                      "(πT=" + str(priors[0]) + "; λ=" + str(x) + ") with " + named_preprocess + " features...")
                train_single_quadratic_logistic_regression_model(priors, dict_preprocess_list[named_preprocess],
                                                                 DTR, LTR, DTE, LTE, x)
                print("========================================================")
    print("================== End QLR Training ====================")

    plotting_train_for_logistic_regression()
    plotting_train_for_quadratic_logistic_regression()