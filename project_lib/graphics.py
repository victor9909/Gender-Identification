import matplotlib.pyplot as plt
from project_lib.preprocessing import *
from project_lib.validation_utils import *
from project_lib.utils import *


def __plot_heatmaps_for_all_classes(D: np.ndarray) -> None:
    heatmap = np.zeros((D.shape[0], D.shape[0]))
    for f1 in range(D.shape[0]):
        for f2 in range(D.shape[0]):
            if f2 <= f1:
                heatmap[f1][f2] = abs(sp.stats.pearsonr(D[f1, :], D[f2, :])[0])
                heatmap[f2][f1] = heatmap[f1][f2]
    plt.figure()
    plt.xticks(np.arange(0, D.shape[0]), np.arange(1, D.shape[0] + 1))
    plt.yticks(np.arange(0, D.shape[0]), np.arange(1, D.shape[0] + 1))
    plt.suptitle('Heatmap of Pearson Correlation Coefficient for both classes')
    plt.imshow(heatmap, cmap="Greys")
    plt.colorbar()
    plt.savefig("figures/heatmaps/AllClasses.png")
    plt.close()


def plot_heatmaps(D: np.ndarray, L: np.ndarray, labels: dict, colors: dict) -> None:
    assert (len(labels.keys()) == np.unique(L).size)
    assert (len(colors.keys()) == np.unique(L).size)
    create_folder_if_not_exist("figures/heatmaps/")

    for i in range(np.unique(L).size):
        descr = labels[i]
        color = colors[i]
        data = D[:, L == i]

        heatmap = np.zeros((data.shape[0], data.shape[0]))
        for f1 in range(data.shape[0]):
            for f2 in range(data.shape[0]):
                if f2 <= f1:
                    heatmap[f1][f2] = abs(sp.stats.pearsonr(data[f1, :], data[f2, :])[0])
                    heatmap[f2][f1] = heatmap[f1][f2]
        plt.figure()
        plt.xticks(np.arange(0, data.shape[0]), np.arange(1, data.shape[0] + 1))
        plt.yticks(np.arange(0, data.shape[0]), np.arange(1, data.shape[0] + 1))
        plt.suptitle('Heatmap of Pearson Correlation Coefficient for ' + descr)
        plt.imshow(heatmap, cmap=color)
        plt.colorbar()
        plt.savefig("figures/heatmaps/" + descr)
        plt.close()

        __plot_heatmaps_for_all_classes(D)


def __apply_prep_step(D: np.ndarray, L: np.ndarray, pre_step: Preprocessing, m: int = None):
    mean = compute_mean(D)
    if pre_step.name == Preprocessing.LDA.name:
        prep = Lda(np.unique(L).size - 1)
        prep.set_attributes(D, L)
        dprop, _ = prep.process()
        return dprop
    elif pre_step.name == Preprocessing.PCA.name:
        prep = Pca(m)
        prep.set_attributes(D - mean, L)
        dprop, _ = prep.process()
        return dprop
    elif pre_step.name == Preprocessing.ZSCORE.name:
        prep = Zscore()
        prep.set_attributes(D - mean, L)
        dprop, _ = prep.process()
        return dprop
    else:
        return D - mean


def plot_fraction_explained_variance_pca(DTR, LTR):
    create_folder_if_not_exist("figures/explained_variance/")
    pca = Pca(DTR.shape[0] + 1)
    pca.set_attributes(DTR, LTR)
    pca.process()
    n_dimensions = pca.eigen_value.size
    sorted_eigenvalues = pca.eigen_value[::-1]
    total_variance = np.sum(sorted_eigenvalues)
    explained_variance_ratio = np.cumsum(sorted_eigenvalues / total_variance)
    plt.plot(range(1, n_dimensions + 1), explained_variance_ratio, marker='o')
    plt.xlabel('Number of dimensions')
    plt.ylabel('Fraction of explained variance')
    plt.savefig("figures/explained_variance/explained_variance.png")
    plt.close()


def plot_histograms(D: np.ndarray, L: np.ndarray, pre_step: Preprocessing, labels: dict, m = None) -> None:
    dprop = __apply_prep_step(D, L, pre_step, m)
    # Check if the number of labels is equal to entries in dict
    assert (len(labels.keys()) == np.unique(L).size)
    if pre_step is Preprocessing.PCA:
        create_folder_if_not_exist("figures/histograms/" + pre_step.name.lower() + str(m))
    else:
        create_folder_if_not_exist("figures/histograms/" + pre_step.name.lower())
    for i in range(dprop.shape[0]):
        plt.figure()
        plt.xlabel('Feature ' + str(i + 1))
        for j in range(np.unique(L).size):
            plt.hist(dprop[:, L == j][i, :], bins=70, density=True, alpha=0.4, label=labels[j], linewidth=1.0,
                     edgecolor='black')
        plt.legend()
        plt.tight_layout()
        if pre_step is Preprocessing.PCA:
            plt.savefig('figures/histograms/' + pre_step.name.lower() + str(m) + '/Feature_' + str(i + 1) + '.png')
        else:
            plt.savefig('figures/histograms/' + pre_step.name.lower() + '/Feature_' + str(i + 1) + '.png')
        plt.close()


def plot_histograms_dataset(D: np.ndarray, L: np.ndarray, pre_step: Preprocessing, labels: dict) -> None:

    dprop = __apply_prep_step(D, L, pre_step)

    # Check if the number of labels is equal to entries in dict
    assert (len(labels.keys()) == np.unique(L).size)

    plt.hist(dprop[:, L == 0].ravel(), density=True, bins=70, alpha=0.4, label=labels[0], linewidth=1.0, edgecolor='black')
    plt.hist(dprop[:, L == 1].ravel(), density=True, bins=70, alpha=0.4, label=labels[0], linewidth=1.0, edgecolor='black')
    plt.legend()
    plt.tight_layout()
    create_folder_if_not_exist('figures/histograms/' + pre_step.name.lower())
    plt.savefig('figures/histograms/' + pre_step.name.lower() + "/datasets.png")
    plt.close()


def plot_bayes_error(score, model: Model, title: str, labels=None, is_train=True):

    plt.figure()
    score = score.ravel()
    n_points = 100
    eff_prior = np.linspace(-4, 4, n_points)
    dcf = np.zeros(n_points)
    min_dcf = np.zeros(n_points)
    plt.title(model.name + " " + title)
    if labels is None:
        labels = load_labels_for_score()

    for (idx, p) in enumerate(eff_prior):
        pi = 1 / (1 + np.exp(-p))
        dcf[idx] = compute_NDCF(score, labels, pi, 1, 1)
        min_dcf[idx] = compute_minimum_NDCF(score, labels, pi, 1, 1)[0]
    plt.plot(eff_prior, dcf, label="actDCF", color="r")
    plt.plot(eff_prior, min_dcf, ':', label="minDCF", color="r")

    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    plt.ylabel("DCF")
    plt.xlabel("$log \\frac{ \\tilde{\pi}}{1-\\tilde{\pi}}$")
    plt.legend()
    if is_train:
        create_folder_if_not_exist("figures/models/train/" + model.name)
        plt.savefig("figures/models/train/" + model.name + "/" + model.name + "_" + title + "_" + "bayes_error.png")
    else:
        create_folder_if_not_exist("figures/models/test/" + model.name)
        plt.savefig("figures/models/test/" + model.name + "/" + model.name + "_" + title + "_" + "bayes_error.png")
    plt.close()


def plot_bayes_error_more_models(score_list: list, model: list, colors, desc: str, labels=None, is_train=True):
    plt.figure()
    for (idx_, score) in enumerate(score_list):
        score = score.ravel()
        n_points = 100
        eff_prior = np.linspace(-4, 4, n_points)
        dcf = np.zeros(n_points)
        min_dcf = np.zeros(n_points)
        if labels is None:
            labels = load_labels_for_score()

        for (idx, p) in enumerate(eff_prior):
            pi = 1 / (1 + np.exp(-p))
            dcf[idx] = compute_NDCF(score, labels, pi, 1, 1)
            min_dcf[idx] = compute_minimum_NDCF(score, labels, pi, 1, 1)[0]
        plt.plot(eff_prior, dcf, label="actDCF - " + model[idx_], color=colors[idx_])
        print(idx_)
        plt.plot(eff_prior, min_dcf, ':', label="minDCF - " + model[idx_], color=colors[idx_])

    plt.ylim([0, 1])
    plt.title("Bayes Error Plot: " + ", ".join(model) + " - " + desc)
    plt.xlim([-4, 4])
    plt.ylabel("DCF")
    plt.xlabel("$log \\frac{ \\tilde{\pi}}{1-\\tilde{\pi}}$")
    plt.legend()

    if is_train:
        create_folder_if_not_exist("figures/models/train/mix")
        plt.savefig("figures/models/train/mix/" + "Bayes Error Plot: " + ", ".join(model) + " - " + desc)
    else:
        create_folder_if_not_exist("figures/models/test/mix")
        plt.savefig("figures/models/test/mix/" + "Bayes Error Plot: " + ", ".join(model) + " - " + desc)
    plt.close()


def plot_det(llr: list, L: np.array, labels: list, colors, file_name: str):

    for (idx, scores) in enumerate(llr):
        fpr, tpr = compute_det_points(scores, L)
        plt.plot(fpr, tpr, color=colors[idx], label=labels[idx])

    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    create_folder_if_not_exist("figures/models/test/det")
    plt.savefig("figures/models/test/det/" + file_name)
    plt.close()


def plot_scatter(DTR, LTR):
    create_folder_if_not_exist("figures/scatter/")
    idx = 0
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[0]):
            if i != j:
                plt.figure()
                plt.scatter(DTR[i, LTR == 0], DTR[j, LTR == 0], label="Male")
                plt.scatter(DTR[i, LTR == 1], DTR[j, LTR == 1], label="Female")
                plt.legend()
                plt.xlabel("Feature " + str(i))
                plt.ylabel("Feature " + str(j))
                plt.savefig("figures/scatter/scatter_" + str(idx))
                idx += 1
                plt.close()


def plot_roc(array: list, filename: str):
    for x in array:
        tpr, fpr = compute_roc_points(x[0], x[1])
        plt.plot(fpr, tpr, color=np.random.rand(len(array)), label=x[2])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.legend()
    plt.savefig('data/figures/roc/' + filename)
    plt.show()


def plot_min_cdf_error_discriminative(model, min_dcf_01, min_dcf_05, min_dcf_09, preprocess_descr):

    lambda_values = np.logspace(-5, 5, num=51)
    plt.figure()
    plt.title(preprocess_descr)
    plt.xlabel('$lambda$')
    plt.xscale('log')
    plt.ylabel("minDCF")

    plt.plot(lambda_values, min_dcf_01, label="minDCF($\\tilde{\pi} = 0.1$)")
    plt.plot(lambda_values, min_dcf_05, label="minDCF($\\tilde{\pi} = 0.5$)")
    plt.plot(lambda_values, min_dcf_09, label="minDCF($\\tilde{\pi} = 0.9$)")
    plt.xlim(lambda_values[0], lambda_values[-1])
    plt.legend()

    create_folder_if_not_exist("figures/models/train/" + model.name)
    plt.savefig("figures/models/train/" + model.name + "/" + preprocess_descr + ".png")
    plt.close()


def plot_min_cdf_error_discriminative_test(model, min_cdf_train, min_cdf_test, preprocess_descr):

    lambda_values = np.logspace(-5, 5, num=51)
    plt.figure()
    plt.title(preprocess_descr)
    plt.xlabel('$lambda$')
    plt.xscale('log')
    plt.ylabel("minDCF")

    plt.plot(lambda_values, min_cdf_train[0], label="minDCF($\\tilde{\pi} = 0.1$) [Val]", linestyle='dotted', color="b")
    plt.plot(lambda_values, min_cdf_train[1], label="minDCF($\\tilde{\pi} = 0.5$) [Val]", linestyle='dotted', color="r")
    plt.plot(lambda_values, min_cdf_train[2], label="minDCF($\\tilde{\pi} = 0.9$) [Val]", linestyle='dotted', color="g")

    plt.plot(lambda_values, min_cdf_test[0], label="minDCF($\\tilde{\pi} = 0.1$) [Eval]", color="b")
    plt.plot(lambda_values, min_cdf_test[1], label="minDCF($\\tilde{\pi} = 0.5$) [Eval]", color="r")
    plt.plot(lambda_values, min_cdf_test[2], label="minDCF($\\tilde{\pi} = 0.9$) [Eval]", color="g")

    plt.xlim(lambda_values[0], lambda_values[-1])
    plt.legend()

    create_folder_if_not_exist("figures/models/test/" + model.name)
    plt.savefig("figures/models/test/" + model.name + "/" + preprocess_descr + ".png")
    plt.close()


def plot_min_cdf_error_no_prob(model, min_dcf_01, min_dcf_05, min_dcf_09, preprocess_descr):

    plt.figure()
    plt.title(preprocess_descr)
    plt.xlabel('$C$')
    c_values = np.logspace(-5, 5, num=31)

    plt.plot(c_values, min_dcf_01, label="minDCF($\\tilde{\pi} = 0.1$)")
    plt.plot(c_values, min_dcf_05, label="minDCF($\\tilde{\pi} = 0.5$)")
    plt.plot(c_values, min_dcf_09, label="minDCF($\\tilde{\pi} = 0.9$)")

    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.xlim(c_values[0], c_values[-1])
    plt.legend()

    create_folder_if_not_exist("figures/models/train/" + model.name)
    plt.savefig("figures/models/train/" + model.name + "/" + preprocess_descr + ".png")
    plt.close()


def plot_min_cdf_error_no_prob_test(model, scores_train, scores_test, preprocess_descr):

    plt.figure()
    plt.title(preprocess_descr)
    plt.xlabel('$C$')
    c_values = np.logspace(-5, 5, num=31)

    plt.plot(c_values, scores_train[0], label="minDCF($\\tilde{\pi} = 0.1$) [Val]", linestyle='dotted', color="b")
    plt.plot(c_values, scores_train[1], label="minDCF($\\tilde{\pi} = 0.5$) [Val]", linestyle='dotted', color="r")
    plt.plot(c_values, scores_train[2], label="minDCF($\\tilde{\pi} = 0.9$) [Val]", linestyle='dotted', color="g")

    plt.plot(c_values, scores_test[0], label="minDCF($\\tilde{\pi} = 0.1$) [Eval]", color="b")
    plt.plot(c_values, scores_test[1], label="minDCF($\\tilde{\pi} = 0.5$) [Eval]", color="r")
    plt.plot(c_values, scores_test[2], label="minDCF($\\tilde{\pi} = 0.9$) [Eval]",color="g")

    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.xlim(c_values[0], c_values[-1])
    plt.legend()

    create_folder_if_not_exist("figures/models/test/" + model.name)
    plt.savefig("figures/models/test/" + model.name + "/" + preprocess_descr + ".png")
    plt.close()


def plot_min_cdf_error_no_prob_radial_based_svm_gamma_values(min_dcf_g_1, min_dcf_g_2, min_dcf_g_3, preprocess_descr):

    plt.figure()
    plt.title(preprocess_descr)
    plt.xlabel('$C$')
    c_values = np.logspace(-5, 5, num=31)

    plt.plot(c_values, min_dcf_g_1, label="$log\gamma = -1$")
    plt.plot(c_values, min_dcf_g_2, label="$log\gamma = -2$")
    plt.plot(c_values, min_dcf_g_3, label="$log\gamma = -3$")

    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.xlim(c_values[0], c_values[-1])
    plt.legend()
    create_folder_if_not_exist("figures/models/train/")
    plt.savefig("figures/models/train/" + Model.RadialBasedSVM.name + "/" + preprocess_descr + ".png")
    plt.close()


def plot_min_cdf_error_no_prob_radial_based_svm_gamma_values_test(min_dcf_train, min_dcf_test, preprocess_descr):

    plt.figure()
    plt.title(preprocess_descr)
    plt.xlabel('$C$')
    c_values = np.logspace(-5, 5, num=31)

    plt.plot(c_values, min_dcf_train[0], label="$log\gamma = -1$ [Val]", linestyle='dotted', color="b")
    plt.plot(c_values, min_dcf_train[1], label="$log\gamma = -2$ [Val]", linestyle='dotted', color="r")
    plt.plot(c_values, min_dcf_train[2], label="$log\gamma = -3$ [Val]", linestyle='dotted', color="g")

    plt.plot(c_values, min_dcf_test[0], label="$log\gamma = -1$ [Eval]",  color="b")
    plt.plot(c_values, min_dcf_test[1], label="$log\gamma = -2$ [Eval]", color="r")
    plt.plot(c_values, min_dcf_test[2], label="$log\gamma = -3$ [Eval]", color="g")

    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.xlim(c_values[0], c_values[-1])
    plt.legend()
    create_folder_if_not_exist("figures/models/test/")
    plt.savefig("figures/models/test/" + Model.RadialBasedSVM.name + "/" + preprocess_descr + ".png")
    plt.close()


def plot_min_cdf_error_gaussian_mixture_models(model, first_dcf, second_dcf, first_label, second_label):

    plt.figure()
    plt.title(model.name)
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    iterations = range(7)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(x_axis + 0.00, first_dcf, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label=first_label)
    plt.bar(x_axis + 0.25, second_dcf, width=0.25, linewidth=1.0, edgecolor='black', color="Orange", label=second_label)

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    create_folder_if_not_exist("figures/models/train/" + model.name)
    plt.savefig("figures/models/train/" + model.name + "/" + first_label + "_" + second_label + ".png")
    plt.close()


def plot_min_cdf_error_gaussian_mixture_models_test(model, dcf_train, dcf_test, labels_train, labels_eval):

    plt.figure()
    plt.title(model.name)
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    iterations = range(7)
    x_axis = np.arange(len(iterations)) * 1.25
    bounds = np.array(iterations)
    plt.bar(x_axis + 0.00, dcf_train[0], width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label=labels_train[0], linestyle="--", hatch="//")
    plt.bar(x_axis + 0.25, dcf_test[0], width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label=labels_eval[0])

    plt.bar(x_axis + 0.50, dcf_train[1], width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label=labels_train[1], linestyle="--", hatch="//")

    plt.bar(x_axis + 0.75, dcf_test[1], width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label=labels_eval[1])

    plt.xticks([r * 1.25 + 0.375 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    create_folder_if_not_exist("figures/models/test/" + model.name)
    plt.savefig("figures/models/test/" + model.name + "/" + labels_train[0] + "_" + labels_train[1] + ".png")
    plt.close()


def compute_std_minCDF_plot_for_model(model: Model, preprocess: list, param_model: dict, is_train=True, param_plot=None):

    prepros_desc = "_".join([str(x).upper() for x in preprocess])
    if len(preprocess) == 0:
        prepros_desc = "RAW"

    assert not is_gaussian_model(model)

    folder_name = ("scores_train" if is_train else "scores_test") + "/" + model.name + "/" + prepros_desc

    if is_discriminative_model(model):
        if is_train:
            min_dcf_01, min_dcf_05, min_dcf_09 = compute_minCDF_for_discriminative_models(folder_name, param_model["prior_t"], is_train, False)
            plot_min_cdf_error_discriminative(model, min_dcf_01, min_dcf_05, min_dcf_09,
                                              prepros_desc + " prior_t" + str(param_model["prior_t"]))
        else:
            min_dcf_01, min_dcf_01_test, min_dcf_05, min_dcf_05_test, min_dcf_09, min_dcf_09_test = compute_minCDF_for_discriminative_models(folder_name, param_model["prior_t"], is_train, False)
            plot_min_cdf_error_discriminative_test(model, [min_dcf_01, min_dcf_05, min_dcf_09], [min_dcf_01_test, min_dcf_05_test, min_dcf_09_test],
                                              prepros_desc + " prior_t" + str(param_model["prior_t"]))
    elif is_svm_models(model):
        gamma = None
        try:
            gamma = param_model["gamma"]
        except KeyError:
            pass
        if is_train:
            min_dcf_01, min_dcf_05, min_dcf_09 = compute_minCDF_for_svm_models(model, folder_name, param_model["prior_t"], gamma, is_train, False)
            plot_min_cdf_error_no_prob(model, min_dcf_01, min_dcf_05, min_dcf_09, prepros_desc + " prior_t" + str(param_model["prior_t"]) if model is not Model.RadialBasedSVM else prepros_desc + " g: " + str(gamma) + " prior_t" + str(param_model["prior_t"]))
        else:
            min_dcf_01, min_dcf_01_test, min_dcf_05, min_dcf_05_test, min_dcf_09, min_dcf_09_test = compute_minCDF_for_svm_models(model, folder_name,
                                                                               param_model["prior_t"], gamma, is_train,
                                                                               False)
            plot_min_cdf_error_no_prob_test(model, [min_dcf_01, min_dcf_05, min_dcf_09], [min_dcf_01_test, min_dcf_05_test, min_dcf_09_test], prepros_desc + " prior_t" + str(
                param_model["prior_t"]) if model is not Model.RadialBasedSVM else prepros_desc + " g: " + str(
                gamma) + " prior_t" + str(param_model["prior_t"]))
        if model is Model.RadialBasedSVM:
            if is_train:
                min_dcf_01_g01, min_dcf_05_g01, min_dcf_09_g01 = compute_minCDF_for_svm_models(model, folder_name, param_model["prior_t"], 0.1, is_train, False)
                min_dcf_01_g02, min_dcf_05_g02, min_dcf_09_g02 = compute_minCDF_for_svm_models(model, folder_name, param_model["prior_t"], 0.01, is_train, False)
                min_dcf_01_g03, min_dcf_05_g03, min_dcf_09_g03 = compute_minCDF_for_svm_models(model, folder_name, param_model["prior_t"], 0.001, is_train, False)
                plot_min_cdf_error_no_prob_radial_based_svm_gamma_values(min_dcf_01_g01, min_dcf_01_g02, min_dcf_01_g03, prepros_desc  + " eff_prior: 0.1")
                plot_min_cdf_error_no_prob_radial_based_svm_gamma_values(min_dcf_05_g01, min_dcf_05_g02, min_dcf_05_g03,  prepros_desc + " eff_prior: 0.5")
                plot_min_cdf_error_no_prob_radial_based_svm_gamma_values(min_dcf_09_g01, min_dcf_09_g02, min_dcf_09_g03,  prepros_desc + " eff_prior: 0.9")
            else:

                min_dcf_01_g01, min_dcf_01_g01_test, min_dcf_05_g01, min_dcf_05_g01_test, min_dcf_09_g01, min_dcf_09_g01_test = compute_minCDF_for_svm_models(model, folder_name,
                                                                                               param_model["prior_t"],
                                                                                               0.1, is_train, False)
                min_dcf_01_g02, min_dcf_01_g02_test, min_dcf_05_g02, min_dcf_05_g02_test, min_dcf_09_g02, min_dcf_09_g02_test = compute_minCDF_for_svm_models(model, folder_name,
                                                                                               param_model["prior_t"],
                                                                                               0.01, is_train, False)
                min_dcf_01_g03, min_dcf_01_g03_test, min_dcf_05_g03, min_dcf_05_g03_test, min_dcf_09_g03, min_dcf_09_g03_test = compute_minCDF_for_svm_models(model, folder_name,
                                                                                               param_model["prior_t"],
                                                                                               0.001, is_train, False)
                plot_min_cdf_error_no_prob_radial_based_svm_gamma_values_test([min_dcf_01_g01, min_dcf_01_g02, min_dcf_01_g03], [min_dcf_01_g01_test, min_dcf_01_g02_test, min_dcf_01_g03_test],
                                                                         prepros_desc + " eff_prior: 0.1")
                plot_min_cdf_error_no_prob_radial_based_svm_gamma_values_test([min_dcf_05_g01, min_dcf_05_g02, min_dcf_05_g03], [min_dcf_05_g01_test, min_dcf_05_g02_test, min_dcf_05_g03_test],
                                                                         prepros_desc + " eff_prior: 0.5")
                plot_min_cdf_error_no_prob_radial_based_svm_gamma_values_test([min_dcf_09_g01, min_dcf_09_g02, min_dcf_09_g03], [min_dcf_09_g01_test, min_dcf_09_g02_test, min_dcf_09_g03_test],
                                                                         prepros_desc + " eff_prior: 0.9")
        else:
            pass
    else:
        if is_train:
            folder_name_first = ("scores_train" if is_train else "scores_test") + "/" + model.name + "/" + param_plot[0].upper()
            _, first_dcf, _ = compute_minCDF_for_gaussian_mixture_models(folder_name_first, is_train, False)
            folder_name_second = ("scores_train" if is_train else "scores_test") + "/" + model.name + "/" + param_plot[1].upper()
            _, second_dcf, _ = compute_minCDF_for_gaussian_mixture_models(folder_name_second, is_train, False)
            plot_min_cdf_error_gaussian_mixture_models(model, first_dcf, second_dcf, param_plot[0], param_plot[1])
        else:
            folder_name_first = ("scores_train" if is_train else "scores_test") + "/" + model.name + "/" + param_plot[0].upper()
            _,_, first_dcf, first_dcf_test, _, _ = compute_minCDF_for_gaussian_mixture_models(folder_name_first, is_train, False)
            folder_name_second = ("scores_train" if is_train else "scores_test") + "/" + model.name + "/" + param_plot[1].upper()
            _,_,  second_dcf, second_dcf_test, _, _ = compute_minCDF_for_gaussian_mixture_models(folder_name_second, is_train, False)
            plot_min_cdf_error_gaussian_mixture_models_test(model, [first_dcf, second_dcf], [first_dcf_test, second_dcf_test], [
                str(param_plot[0] + " [Val]"),
                str(param_plot[1] + " [Val]")
            ], [
                str(param_plot[0] + " [Eval]"),
                str(param_plot[1] + " [Eval]")
            ])
