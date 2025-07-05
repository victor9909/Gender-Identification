from project_lib.models.Model import *
from project_lib.utils import *


def logpdf_GAU_ND(X: np.ndarray, mu: np.ndarray, C: np.ndarray) -> np.ndarray:
    M = X.shape[0]
    _, det_C = np.linalg.slogdet(C)
    inv_C = np.linalg.inv(C)
    density_array = -0.5 * M * np.log(2 * np.pi) - 0.5 * det_C
    density_array = density_array - 0.5 * ((X - mu) * np.dot(inv_C, (X - mu))).sum(0)
    return density_array


def compute_gaussian_scores(D: np.ndarray, mu_cn: np.ndarray, C_cn: np.ndarray):
    SJoint = np.zeros((len(mu_cn), D.shape[1]))
    for i in range(len(mu_cn)):
        SJoint[i:i + 1, :] = np.exp(logpdf_GAU_ND(D, mu_cn[i], C_cn[i]))
    llr = np.zeros(SJoint.shape[1])
    for i in range(SJoint.shape[1]):
        llr[i] = np.log(SJoint[1, i] / SJoint[0, i])
    return llr


class MVG(Models):

    def __init__(self):
        super().__init__([])
        self.cov_matrix = None
        self.means = None
        self.SPost = None

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        super().set_attributes(DTR, LTR, DTE, LTE)

    def train(self):
        self.means = computes_mean_classes(self.DTR, self.LTR)
        self.cov_matrix = compute_covariance_matrices_for_classes(self.DTR, self.LTR)

    def compute_scores(self):
        self.scores = compute_gaussian_scores(self.DTE, self.means, self.cov_matrix)

    def __str__(self):
        return "MVG_"


class NBG(Models):

    def __init__(self):
        super().__init__([])
        self.cov_matrix = None
        self.means = None
        self.SPost = None

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        super().set_attributes(DTR, LTR, DTE, LTE)

    def train(self):
        self.means = computes_mean_classes(self.DTR, self.LTR)
        cov_matrix = compute_covariance_matrices_for_classes(self.DTR, self.LTR)
        classes = np.unique(self.LTR).size
        self.cov_matrix = [cov_matrix[i] * np.identity(self.DTR.shape[0]) for i in range(classes)]

    def compute_scores(self):
        self.scores = compute_gaussian_scores(self.DTE, self.means, self.cov_matrix)

    def __str__(self):
        return "NBG_"


class TNB(Models):

    def __init__(self):
        super().__init__([])
        self.cov_matrix = None
        self.means = None
        self.SPost = None

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        super().set_attributes(DTR, LTR, DTE, LTE)

    def train(self):
        self.means = computes_mean_classes(self.DTR, self.LTR)
        cov_matrix = compute_covariance_matrices_for_classes(self.DTR, self.LTR)
        matrices = [cov_matrix[i] * np.eye(self.DTR.shape[0]) for i in range(len(cov_matrix))]
        C = np.zeros((matrices[0].shape[0], matrices[0].shape[0]))
        for i in range(len(matrices)):
            C += (self.LTR == i).sum() * matrices[i]
        self.cov_matrix = [C / self.DTR.shape[1] for _ in range(len(cov_matrix))]

    def compute_scores(self):
        self.scores = compute_gaussian_scores(self.DTE, self.means, self.cov_matrix)

    def __str__(self):
        return "TNB_"


class TMVG(Models):

    def __init__(self):
        super().__init__([])
        self.cov_matrix = None
        self.means = None
        self.SPost = None

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        super().set_attributes(DTR, LTR, DTE, LTE)

    def train(self):
        self.means = computes_mean_classes(self.DTR, self.LTR)
        covariances = compute_covariance_matrices_for_classes(self.DTR, self.LTR)
        C = np.zeros((covariances[0].shape[0], covariances[0].shape[0]))
        for i in range(len(covariances)):
            C += (self.LTR == i).sum() * covariances[i]
        self.cov_matrix = [C / self.DTR.shape[1] for _ in range(len(covariances))]

    def compute_scores(self):
        self.scores = compute_gaussian_scores(self.DTE, self.means, self.cov_matrix)

    def __str__(self):
        return "TMVG_"
