from itertools import compress

from distributed.deploy.old_ssh import bcolors

from project_lib.models.discriminative_models import LR
from project_lib.models.Model import *
from project_lib.utils import *


def debug_print_information(model, labels, calibration):
    predicted_labels = np.where(model.scores > 0, 1, 0)
    err = (1 - (labels == predicted_labels).sum() / labels.size) * 100
    print("Error rate for this training is " + bcolors.BOLD + str(round(err, 2)) + "%" + bcolors.ENDC)
    cost_0_5 = str(round(compute_minimum_NDCF(model.scores, labels, 0.5, 1, 1)[0], 3))
    cost_0_1 = str(round(compute_minimum_NDCF(model.scores, labels, 0.1, 1, 1)[0], 3))
    cost_0_9 = str(round(compute_minimum_NDCF(model.scores, labels, 0.9, 1, 1)[0], 3))
    print("minDCF with π=0.5 " + bcolors.BOLD + cost_0_5 + bcolors.ENDC)
    print("minDCF with π=0.1 " + bcolors.BOLD + cost_0_1 + bcolors.ENDC)
    print("minDCF with π=0.9 " + bcolors.BOLD + cost_0_9 + bcolors.ENDC)
    if calibration:
        cost_0_5_cal = str(round(compute_NDCF(model.scores, labels, 0.5, 1, 1), 3))
        cost_0_1_cal = str(round(compute_NDCF(model.scores, labels, 0.1, 1, 1), 3))
        cost_0_9_cal = str(round(compute_NDCF(model.scores, labels, 0.9, 1, 1), 3))
        print("actDCF with π=0.5 " + bcolors.BOLD + cost_0_5_cal + bcolors.ENDC)
        print("actDCF with π=0.1 " + bcolors.BOLD + cost_0_1_cal + bcolors.ENDC)
        print("actDCF with π=0.9 " + bcolors.BOLD + cost_0_9_cal + bcolors.ENDC)


class Pipe:

    def __init__(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray, preprocessing: list,
                 model: Models):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.preprocessing = preprocessing
        self.model = model

    def __define_folder_name(self, is_train) -> (str, str):

        event = "train" if is_train else "test"
        prepros_desc = "_".join([str(x) for x in self.preprocessing])
        if len(self.preprocessing) == 0:
            prepros_desc = "RAW"

        desc = "scores_" + event + "/" + str(self.model).replace("_", "") + "/" + str(prepros_desc) + "/"
        priors_desc = ""
        if len(self.model.priors) != 0:
            priors_desc = "_C0_" + str(self.model.priors[0]) + "_C1_" + str(self.model.priors[1])
        if hasattr(self.model, 'params'):
            for x in self.model.params:
                priors_desc += "_" + x[0] + "_" + str(x[1])

        return desc, desc + str(self.model) + prepros_desc + priors_desc

    def make_train_with_K_fold(self, K=5, seed=27, calibration=False, fusion=False, model_calibrated=None, fusion_desc=None):

        if calibration:
            assert not fusion and model_calibrated is not None
        if fusion:
            assert not calibration and fusion_desc is not None
        # Setting up all data needed
        D, L, idx = split_data(self.DTR, self.LTR, K, seed)
        mask = np.array([False for _ in range(K)])
        scores = np.zeros(self.LTR.shape[0])
        n_folds = self.LTR.shape[0] // K
        labels_training = self.LTR[idx]

        # Setting up the folder name
        folder_descr, path = self.__define_folder_name(True)

        for i in range(K):

            mask[i] = True

            DTE = np.array(list(compress(D, mask))).reshape(-1, D[0].shape[1])
            DTR = np.hstack(np.array(list(compress(D, ~mask))))
            LTE = np.array(list(compress(L, mask))).ravel()
            LTR = np.hstack(np.array(list(compress(L, ~mask))))

            # Apply Preprocessing Step on Data
            for preprocess_step in self.preprocessing:
                preprocess_step.set_attributes(DTR, LTR, DTE, LTE)
                DTR, DTE = preprocess_step.process()

            # Apply the model selected for the pipe with training and save the score
            self.model.set_attributes(DTR, LTR, DTE, LTE)
            self.model.train()
            self.model.compute_scores()
            scores[i * n_folds: (i + 1) * n_folds] = self.model.scores
            mask[i] = False
        self.model.scores = scores

        # Print some debug information
        debug_print_information(self.model, labels_training, calibration)

        # Create the folder if not exists and save the score associated to this train
        if not calibration and not fusion:
            create_folder_if_not_exist(folder_descr)
            np.save(path, scores)

        if calibration and not fusion:
            create_folder_if_not_exist("scores_train/calibration")
            np.save("scores_train/calibration/scores_calibrated_LR_pt_" + str(self.model.priors[1])+ "_" +
                                       str(model_calibrated) + ".npy", scores)

        if not calibration and fusion:
            create_folder_if_not_exist("scores_train/fusion")
            np.save("scores_train/fusion/scores_fused_LR_pt_" + str(self.model.priors[1]) + "_" +
                    str(fusion_desc) + ".npy", scores)

    def make_evaluation(self, calibration=False, fusion=False, model_calibrated=None, fusion_desc=None):

        if calibration:
            assert not fusion and model_calibrated is not None
        if fusion:
            assert not calibration and fusion_desc is not None

        # Setting up the folder name
        folder_descr, path = self.__define_folder_name(False)

        # Apply Preprocessing Step on Data
        for preprocess_step in self.preprocessing:
            preprocess_step.set_attributes(self.DTR, self.LTR, self.DTE, self.LTE)
            self.DTR, self.DTE = preprocess_step.process()

        # Apply the model selected for the pipe with training and save the score
        self.model.set_attributes(self.DTR, self.LTR, self.DTE, self.LTE)
        self.model.train()
        self.model.compute_scores()

        # Print some debug information
        debug_print_information(self.model, self.model.LTE, calibration)

        if calibration and not fusion:
            create_folder_if_not_exist("scores_test/calibration")
            np.save("scores_test/calibration/scores_calibrated_LR_pt_" + str(self.model.priors[1])+ "_" +
                                       str(model_calibrated) + ".npy", self.model.scores)

        if not calibration and fusion:
            create_folder_if_not_exist("scores_test/fusion")
            np.save("scores_test/fusion/scores_fused_LR_pt_" + str(self.model.priors[1]) + "_" +
                    str(fusion_desc) + ".npy", self.model.scores)

        # Create the folder if not exists and save the score associated to this train
        if not calibration and not fusion:
            create_folder_if_not_exist(folder_descr)
            np.save(path, self.model.scores)


def make_calibration(scores, labels, priors, model_descr):
    np.random.seed(27)
    idx = np.random.permutation(labels.size)
    pipe_calibration = Pipe(np.array([scores]), labels[idx], np.array([]), np.array([]), [], LR(priors, 0))
    pipe_calibration.make_train_with_K_fold(calibration=True,  model_calibrated=model_descr)
    calibrated_score = pipe_calibration.model.scores - np.log(priors[1] / priors[0])
    return calibrated_score, (labels[idx])[idx]


def make_eval_calibration(train_scores, labels_train, test_scores, labels_test, priors, model_descr):

    np.random.seed(27)
    idx = np.random.permutation(labels_train.size)
    pipe_calibration = Pipe(v_row(train_scores), labels_train[idx], v_row(test_scores), labels_test, [], LR(priors, 0))
    pipe_calibration.make_evaluation(calibration=True,  model_calibrated=model_descr)
    calibrated_score = pipe_calibration.model.scores - np.log(priors[1] / priors[0])
    return calibrated_score


def make_fusion(scores, labels, priors, fusion_desc):

    np.random.seed(27)
    idx = np.random.permutation(labels.size)
    pipe_fusion = Pipe(np.array(np.vstack(scores)), labels[idx], np.array([]), np.array([]), [], LR(priors, 0))
    pipe_fusion.make_train_with_K_fold(fusion=True, fusion_desc=fusion_desc)

    return pipe_fusion.model.scores, (labels[idx])[idx]


def make_eval_fusion(scores_train, labels_train, scores_test, labels_test, priors, fusion_desc):

    DTR = np.vstack(scores_train)
    np.random.seed(27)
    idx = np.random.permutation(labels_train.size)
    LTR = labels_train[idx]
    DTE = np.vstack(scores_test)
    pipe_fusion = Pipe(DTR, LTR, DTE, labels_test, [], LR(priors, 0))
    pipe_fusion.make_evaluation(fusion=True, fusion_desc=fusion_desc)
    return pipe_fusion.model.scores


