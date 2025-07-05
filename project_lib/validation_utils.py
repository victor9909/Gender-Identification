import numpy as np

from project_lib.utils import *
from project_lib.models.Model import *


def is_gaussian_model(model: Model):
    return model == Model.MVG or model == Model.TMVG or model == Model.TNB or model == Model.NBG


def is_discriminative_model(model: Model):
    return model == Model.LR or model == Model.QLR


def is_svm_models(model: Model):
    return model == Model.SVM or model == Model.PolSVM or model == model.RadialBasedSVM


def is_gaussian_mixture_model(model: Model):
    return model == Model.GMM or model == Model.GMMTied or model == model.GMMDiagonal or model == Model.GMMTiedDiagonal


def compute_minCDF_for_gaussian_models(folder_name, is_train=True, verbose=True):

    dtr, ltr, dte, lte = load_dataset()
    _, _, idx = split_data(dtr, ltr)
    file_path = "_".join(folder_name.split("/")[1:3])
    min_dcf_01, min_dcf_05, min_dcf_09 = [], [], []
    if is_train:
        scores = np.load(folder_name + "/" + file_path + ".npy")
        if verbose:
            print("0.5: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0], 3)))
            print("0.1: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0], 3)))
            print("0.9: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0], 3)))
        else:
            min_dcf_05.append(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0])
            min_dcf_01.append(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0])
            min_dcf_09.append(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0])

        return min_dcf_01, min_dcf_05, min_dcf_09
    else:
        print("Not implemented yet")


def compute_minCDF_for_discriminative_models(folder_name, prior_t, is_train=True, verbose=True):

    # Discriminative Models
    l_values = np.logspace(-5, 5, num=51)
    folder_name_train = ""
    folder_name_test = ""

    dtr, ltr, dte, lte = load_dataset()
    file_path = "_".join(folder_name.split("/")[1:3]) + "_C0_" + str(round(1 - prior_t, 1)) + "_C1_" + str(round(prior_t, 1)) + "_l_"
    _, _, idx = split_data(dtr, ltr)
    min_dcf_01, min_dcf_05, min_dcf_09 = [], [], []
    min_dcf_01_test, min_dcf_05_test, min_dcf_09_test = [], [], []

    if not is_train:
        folder_name_train = "scores_train/" + "/".join(folder_name.split("/")[1:3])
        folder_name_test = "scores_test/" + "/".join(folder_name.split("/")[1:3])

    print("Prior for true class: " + str(prior_t))

    if is_train:
        for l_val in l_values:
            scores = np.load(folder_name + "/" + file_path + str(l_val) + ".npy")
            if verbose:
                print("Value of lambda: " + str(l_val))
                print("0.5: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0], 3)))
                print("0.1: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0], 3)))
                print("0.9: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0], 3)))
            else:
                min_dcf_05.append(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0])
                min_dcf_01.append(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0])
                min_dcf_09.append(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0])
        return min_dcf_01, min_dcf_05, min_dcf_09
    else:
        for l_val in l_values:
            scores_train = np.load(folder_name_train + "/" + file_path + str(l_val) + ".npy")
            scores_test = np.load(folder_name_test + "/" + file_path + str(l_val) + ".npy")

            min_dcf_05.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.5, 1, 1)[0])
            min_dcf_01.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.1, 1, 1)[0])
            min_dcf_09.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.9, 1, 1)[0])

            min_dcf_05_test.append(compute_minimum_NDCF(scores_test, lte, 0.5, 1, 1)[0])
            min_dcf_01_test.append(compute_minimum_NDCF(scores_test, lte, 0.1, 1, 1)[0])
            min_dcf_09_test.append(compute_minimum_NDCF(scores_test, lte, 0.9, 1, 1)[0])

        return min_dcf_01, min_dcf_01_test, min_dcf_05, min_dcf_05_test, min_dcf_09, min_dcf_09_test



def compute_minCDF_for_svm(folder_name, prior_t, is_train=True, verbose=True):

    # No Probabilistic Models
    c_values = np.logspace(-5, 5, num=31)
    folder_name_train = ""
    folder_name_test = ""

    dtr, ltr, dte, lte = load_dataset()
    file_path = "_".join(folder_name.split("/")[1:3]) + "_C0_" + str(round(1 - prior_t, 1)) + "_C1_" + str(str(round(prior_t, 1))) + "_K_1_C_"
    _, _, idx = split_data(dtr, ltr)
    min_dcf_01, min_dcf_05, min_dcf_09 = [], [], []
    min_dcf_01_test, min_dcf_05_test, min_dcf_09_test = [], [], []

    if not is_train:
        folder_name_train = "scores_train/" + "/".join(folder_name.split("/")[1:3])
        folder_name_test = "scores_test/" + "/".join(folder_name.split("/")[1:3])

    print("Prior for true class: " + str(prior_t))

    if is_train:
        for c_val in c_values:
            scores = np.load(folder_name + "/" + file_path + str(c_val) + ".npy")
            if verbose:
                print("Value of C: " + str(c_val))
                print("0.5: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0], 3)))
                print("0.1: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0], 3)))
                print("0.9: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0], 3)))
            else:
                min_dcf_05.append(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0])
                min_dcf_01.append(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0])
                min_dcf_09.append(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0])
        return min_dcf_01, min_dcf_05, min_dcf_09
    else:
        for c_val in c_values:
            scores_train = np.load(folder_name_train + "/" + file_path + str(c_val) + ".npy")
            scores_test = np.load(folder_name_test + "/" + file_path + str(c_val) + ".npy")

            min_dcf_05.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.5, 1, 1)[0])
            min_dcf_01.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.1, 1, 1)[0])
            min_dcf_09.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.9, 1, 1)[0])

            min_dcf_05_test.append(compute_minimum_NDCF(scores_test, lte, 0.5, 1, 1)[0])
            min_dcf_01_test.append(compute_minimum_NDCF(scores_test, lte, 0.1, 1, 1)[0])
            min_dcf_09_test.append(compute_minimum_NDCF(scores_test, lte, 0.9, 1, 1)[0])

        return min_dcf_01, min_dcf_01_test, min_dcf_05, min_dcf_05_test, min_dcf_09, min_dcf_09_test


def compute_minCDF_for_poly_svm(folder_name, prior_t, is_train=True, verbose=True):

    # No Probabilistic Models
    c_values = np.logspace(-5, 5, num=31)
    c = 1
    d = 2
    folder_name_train = ""
    folder_name_test = ""
    min_dcf_01, min_dcf_05, min_dcf_09 = [], [], []
    min_dcf_01_test, min_dcf_05_test, min_dcf_09_test = [], [], []

    if not is_train:
        folder_name_train = "scores_train/" + "/".join(folder_name.split("/")[1:3])
        folder_name_test = "scores_test/" + "/".join(folder_name.split("/")[1:3])

    dtr, ltr, dte, lte = load_dataset()
    file_path = "_".join(folder_name.split("/")[1:3]) + "_C0_" + str(round(1 - prior_t, 1)) + "_C1_" + str(str(round(prior_t, 1))) + "_K_1_C_"
    _, _, idx = split_data(dtr, ltr)

    print("Prior for true class: " + str(prior_t))

    if is_train:
        for c_val in c_values:
            scores = np.load(folder_name + "/" + file_path + str(c_val) + "_c_" + str(c) + "_d_" + str(d) + ".npy")
            if verbose:
                print("Value of C: " + str(c_val))
                print("0.5: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0], 3)))
                print("0.1: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0], 3)))
                print("0.9: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0], 3)))
            else:
                min_dcf_05.append(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0])
                min_dcf_01.append(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0])
                min_dcf_09.append(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0])
        return min_dcf_01, min_dcf_05, min_dcf_09

    else:
        for c_val in c_values:
            scores_train = np.load(folder_name_train + "/" + file_path + str(c_val) + ".npy")
            scores_test = np.load(folder_name_test + "/" + file_path + str(c_val) + ".npy")

            min_dcf_05.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.5, 1, 1)[0])
            min_dcf_01.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.1, 1, 1)[0])
            min_dcf_09.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.9, 1, 1)[0])

            min_dcf_05_test.append(compute_minimum_NDCF(scores_test, lte, 0.5, 1, 1)[0])
            min_dcf_01_test.append(compute_minimum_NDCF(scores_test, lte, 0.1, 1, 1)[0])
            min_dcf_09_test.append(compute_minimum_NDCF(scores_test, lte, 0.9, 1, 1)[0])

        return min_dcf_01, min_dcf_01_test, min_dcf_05, min_dcf_05_test, min_dcf_09, min_dcf_09_test


def compute_minCDF_for_radial_based_svm(folder_name, prior_t, gamma, is_train=True, verbose=True):

    # No Probabilistic Models
    c_values = np.logspace(-5, 5, num=31)
    folder_name_train = ""
    folder_name_test = ""
    min_dcf_01, min_dcf_05, min_dcf_09 = [], [], []
    min_dcf_01_test, min_dcf_05_test, min_dcf_09_test = [], [], []

    if not is_train:
        folder_name_train = "scores_train/" + "/".join(folder_name.split("/")[1:3])
        folder_name_test = "scores_test/" + "/".join(folder_name.split("/")[1:3])

    assert(gamma is not None)

    dtr, ltr, dte, lte = load_dataset()
    file_path = "_".join(folder_name.split("/")[1:3]) + "_C0_" + str(round(1 - prior_t, 1)) + "_C1_" + str(str(round(prior_t, 1))) + "_K_1_C_"
    _, _, idx = split_data(dtr, ltr)
    print("Prior for true class: " + str(prior_t) + " and gamma value: " + str(gamma))

    if is_train:
        for c_val in c_values:
            scores = np.load(folder_name + "/" + file_path + str(c_val) + "_g_" + str(gamma) + ".npy")
            if verbose:
                print("Value of C: " + str(c_val))
                print("0.5: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0], 3)))
                print("0.1: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0], 3)))
                print("0.9: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0], 3)))
            else:
                min_dcf_05.append(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0])
                min_dcf_01.append(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0])
                min_dcf_09.append(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0])
        return min_dcf_01, min_dcf_05, min_dcf_09
    else:
        for c_val in c_values:
            scores_train = np.load(folder_name_train + "/" + file_path + str(c_val) + "_g_" + str(gamma) + ".npy")
            scores_test = np.load(folder_name_test + "/" + file_path + str(c_val) + "_g_" + str(gamma) + ".npy")

            min_dcf_05.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.5, 1, 1)[0])
            min_dcf_01.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.1, 1, 1)[0])
            min_dcf_09.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.9, 1, 1)[0])

            min_dcf_05_test.append(compute_minimum_NDCF(scores_test, lte, 0.5, 1, 1)[0])
            min_dcf_01_test.append(compute_minimum_NDCF(scores_test, lte, 0.1, 1, 1)[0])
            min_dcf_09_test.append(compute_minimum_NDCF(scores_test, lte, 0.9, 1, 1)[0])

        return min_dcf_01, min_dcf_01_test, min_dcf_05, min_dcf_05_test, min_dcf_09, min_dcf_09_test


def compute_minCDF_for_svm_models(svm_model: Model, folder_name, prior_t, gamma=None, is_train=True, verbose=True):

    if svm_model == Model.SVM:
        return compute_minCDF_for_svm(folder_name, prior_t, is_train, verbose)
    elif svm_model == Model.PolSVM:
        return compute_minCDF_for_poly_svm(folder_name, prior_t, is_train, verbose)
    else:
        return compute_minCDF_for_radial_based_svm(folder_name, prior_t, gamma, is_train, verbose)


def compute_minCDF_for_gaussian_mixture_models(folder_name, is_train=True, verbose=True):

    # GMM Models
    iterations = range(7)

    min_dcf_01, min_dcf_05, min_dcf_09 = [], [], []
    min_dcf_01_test, min_dcf_05_test, min_dcf_09_test = [], [], []
    folder_name_train = ""
    folder_name_test = ""

    if not is_train:
        folder_name_train = "scores_train/" + "/".join(folder_name.split("/")[1:3])
        folder_name_test = "scores_test/" + "/".join(folder_name.split("/")[1:3])

    dtr, ltr, dte, lte = load_dataset()
    _, _, idx = split_data(dtr, ltr)
    file_path = "_".join(folder_name.split("/")[1:3]) + "_components_"

    if is_train:
        for i in iterations:
            scores = np.load(folder_name + "/" + file_path + str(2**i) + "_alpha_0.1_psi_0.01.npy")
            if verbose:
                print("Value of component: " + str(2**i))
                print("0.5: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0], 3)))
                print("0.1: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0], 3)))
                print("0.9: " + str(round(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0], 3)))
            else:
                min_dcf_05.append(compute_minimum_NDCF(scores, ltr[idx], 0.5, 1, 1)[0])
                min_dcf_01.append(compute_minimum_NDCF(scores, ltr[idx], 0.1, 1, 1)[0])
                min_dcf_09.append(compute_minimum_NDCF(scores, ltr[idx], 0.9, 1, 1)[0])
        return min_dcf_01, min_dcf_05, min_dcf_09
    else:
        for i in iterations:
            scores_train = np.load(folder_name_train + "/" + file_path + str(2**i) + "_alpha_0.1_psi_0.01.npy")
            scores_test = np.load(folder_name_test + "/" + file_path + str(2**i) + "_alpha_0.1_psi_0.01.npy")

            min_dcf_05.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.5, 1, 1)[0])
            min_dcf_01.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.1, 1, 1)[0])
            min_dcf_09.append(compute_minimum_NDCF(scores_train, ltr[idx], 0.9, 1, 1)[0])

            min_dcf_05_test.append(compute_minimum_NDCF(scores_test, lte, 0.5, 1, 1)[0])
            min_dcf_01_test.append(compute_minimum_NDCF(scores_test, lte, 0.1, 1, 1)[0])
            min_dcf_09_test.append(compute_minimum_NDCF(scores_test, lte, 0.9, 1, 1)[0])

        return min_dcf_01, min_dcf_01_test, min_dcf_05, min_dcf_05_test, min_dcf_09, min_dcf_09_test


def compute_minCDF_for_model(model: Model, preprocess: list, param_model: dict, is_train=True):

    prepros_desc = "_".join([str(x) for x in preprocess])
    if len(preprocess) == 0:
        prepros_desc = "RAW"

    folder_name = ("scores_train" if is_train else "scores_test") + "/" + model.name + "/" + prepros_desc
    if is_gaussian_model(model):
        compute_minCDF_for_gaussian_models(folder_name, is_train)
    elif is_discriminative_model(model):
        compute_minCDF_for_discriminative_models(folder_name, param_model["prior_t"], is_train)
    elif is_svm_models(model):
        gamma = None
        try:
            gamma = param_model["gamma"]
        except KeyError:
            pass
        compute_minCDF_for_svm_models(model, folder_name, param_model["prior_t"], gamma, is_train)
    else:
        compute_minCDF_for_gaussian_mixture_models(folder_name, is_train)


def load_labels_for_score():
    dtr, ltr, dte, lte = load_dataset()
    _, _, idx = split_data(dtr, ltr)
    return ltr[idx]
