from project_lib.models.svm_models import *
from project_lib.models.pipe import *
from project_lib.graphics import compute_std_minCDF_plot_for_model
from project_lib.preprocessing import *


def train_single_linear_svm(DTR, LTR, DTE, LTE, k, c_values, priors, preprocess):

    named_preprocess = str(preprocess)
    if preprocess is None:
        named_preprocess = "RAW"

    for C in c_values:
        print("========================================================")
        print("Training " + str(LinearSvm([], None, None)).replace("_", "") +
              "(πT=" + str(priors[0]) + "; C=" + str(C) + ") with " + named_preprocess + " features...")
        pipe_linear_svm = Pipe(DTR, LTR, DTE, LTE, [preprocess] if preprocess is not None else [], LinearSvm(priors, k, C))
        pipe_linear_svm.make_train_with_K_fold()
        print("========================================================")


def train_single_polynomial_svm(DTR, LTR, DTE, LTE, k, c_values, priors, preprocess, d=2, c=1):

    named_preprocess = str(preprocess)
    if preprocess is None:
        named_preprocess = "RAW"

    for C in c_values:
        print("========================================================")
        print("Training " + str(PolynomialSvm([], None, None, None, None)).replace("_", "") +
              "(πT=" + str(priors[0]) + "; C=" + str(C) + ") with " + named_preprocess + " features...")
        pipe_polynomial_svm = Pipe(DTR, LTR, DTE, LTE, [preprocess] if preprocess is not None else [], PolynomialSvm(priors, k, C, d, c))
        pipe_polynomial_svm.make_train_with_K_fold()
        print("========================================================")


def train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors, preprocess, gamma):

    named_preprocess = str(preprocess)
    if preprocess is None:
        named_preprocess = "RAW"

    for C in c_values:
        print("========================================================")
        print("Training " + str(RadialKernelBasedSvm([], None, None, None)).replace("_", "") +
              "(πT=" + str(priors[0]) + "; γ=" + str(gamma) + "; C=" + str(C) + ") with " + named_preprocess + " features...")
        pipe_radial_kernel_based_svm = Pipe(DTR, LTR, DTE, LTE, [preprocess] if preprocess is not None else [], RadialKernelBasedSvm(priors, k, C, gamma))
        pipe_radial_kernel_based_svm.make_train_with_K_fold()
        print("========================================================")
    pipe_radial_kernel_based_svm = Pipe(DTR, LTR, DTE, LTE, [preprocess] if preprocess is not None else [],
                                        RadialKernelBasedSvm(priors, k, 5, gamma))
    pipe_radial_kernel_based_svm.make_train_with_K_fold()


def train_linear_svm(DTR, LTR, DTE, LTE):

    # Linear SVM Param
    k = 1
    c_values = np.logspace(-5, 5, num=31)
    priors = [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]

    train_single_linear_svm(DTR, LTR, DTE, LTE, k, c_values, priors[0], Zscore())
    train_single_linear_svm(DTR, LTR, DTE, LTE, k, c_values, priors[1], Zscore())
    train_single_linear_svm(DTR, LTR, DTE, LTE, k, c_values, priors[2], Zscore())

    train_single_linear_svm(DTR, LTR, DTE, LTE, k, c_values, priors[0], None)
    train_single_linear_svm(DTR, LTR, DTE, LTE, k, c_values, priors[1], None)
    train_single_linear_svm(DTR, LTR, DTE, LTE, k, c_values, priors[2], None)


def train_polynomial_svm(DTR, LTR, DTE, LTE):

    # Polynomial SVM Param
    k = 1
    d = 2
    c = 1
    c_values = np.logspace(-5, 5, num=31)
    priors = [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]

    train_single_polynomial_svm(DTR, LTR, DTE, LTE, k, c_values, priors[0], Zscore(), d, c)
    train_single_polynomial_svm(DTR, LTR, DTE, LTE, k, c_values, priors[1], Zscore(), d, c)
    train_single_polynomial_svm(DTR, LTR, DTE, LTE, k, c_values, priors[2], Zscore(), d, c)

    train_single_polynomial_svm(DTR, LTR, DTE, LTE, k, c_values, priors[0], None, d, c)
    train_single_polynomial_svm(DTR, LTR, DTE, LTE, k, c_values, priors[1], None, d, c)
    train_single_polynomial_svm(DTR, LTR, DTE, LTE, k, c_values, priors[2], None, d, c)


def train_radial_kernel_based_svm(DTR, LTR, DTE, LTE):

    # Radial Based Kernel SVM Param
    k = 1
    gamma_values = [0.001, 0.01, 0.1]
    c_values = np.logspace(-5, 5, num=31)
    priors = [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]

    # Gamma values 0.001 + ZSCORE
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[0], Zscore(), gamma_values[0])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[1], Zscore(), gamma_values[0])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[2], Zscore(), gamma_values[0])

    # Gamma values 0.01 + ZSCORE
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[0], Zscore(), gamma_values[1])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[1], Zscore(), gamma_values[1])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[2], Zscore(), gamma_values[1])
    #
    # Gamma values 0.1 + ZSCORE
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[0], Zscore(), gamma_values[2])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[1], Zscore(), gamma_values[2])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[2], Zscore(), gamma_values[2])

    # # Gamma values 0.001 + Raw
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[0], None, gamma_values[0])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[1], None, gamma_values[0])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[2], None, gamma_values[0])

    # Gamma values 0.01 + Raw
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[0], None, gamma_values[1])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[1], None, gamma_values[1])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[2], None, gamma_values[1])

    # Gamma values 0.1 + Raw
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[0], None, gamma_values[2])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[1], None, gamma_values[2])
    train_single_radial_kernel_svm(DTR, LTR, DTE, LTE, k, c_values, priors[2], None, gamma_values[2])


def train(DTR, LTR, DTE, LTE):
    # Training linear SVM
    print("================== Start SVM Training ==================")
    train_linear_svm(DTR, LTR, DTE, LTE)
    print("================== End SVM Training ====================")
    # Training polynomial SVM
    print("================= Start QSVM Training ==================")
    train_polynomial_svm(DTR, LTR, DTE, LTE)
    print("================= End QSVM Training ====================")
    # Training Radial Based Kernel SVM
    print("================ Start RBSVM Training =================")
    train_radial_kernel_based_svm(DTR, LTR, DTE, LTE)
    print("================= End RBSVM Training ===================")

    priors = [0.1, 0.5, 0.9]
    for p in priors:
        compute_std_minCDF_plot_for_model(Model.SVM, ["Zscore"], {"prior_t": p})
        compute_std_minCDF_plot_for_model(Model.SVM, [], {"prior_t": p})

        compute_std_minCDF_plot_for_model(Model.PolSVM, ["Zscore"], {"prior_t": p})
        compute_std_minCDF_plot_for_model(Model.PolSVM, [], {"prior_t": p})

        compute_std_minCDF_plot_for_model(Model.RadialBasedSVM, ["Zscore"], {"prior_t": p, "gamma": 0.1})
        compute_std_minCDF_plot_for_model(Model.RadialBasedSVM, ["Zscore"], {"prior_t": p, "gamma": 0.01})
        compute_std_minCDF_plot_for_model(Model.RadialBasedSVM, ["Zscore"], {"prior_t": p, "gamma": 0.001})