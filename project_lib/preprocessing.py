import enum
from abc import abstractmethod
import scipy as sp

from project_lib.utils import *


class Preprocessing(enum.Enum):
    PCA = 0
    LDA = 1
    ZSCORE = 2
    RAW = 3


class Preprocess:

    def __init__(self):
        self.classes = None
        self.M = None
        self.LTE = None
        self.LTR = None
        self.DTE = None
        self.DTR = None

    @abstractmethod
    def process(self):
        pass

    def set_attributes(self, dtr: np.ndarray, dte: np.ndarray, ltr: np.ndarray, lte: np.ndarray):
        self.DTR = dtr
        self.DTE = dte
        self.LTR = ltr
        self.LTE = lte
        self.classes = np.unique(self.LTR).size


class Pca(Preprocess):

    def __init__(self, m: int):
        super().__init__()
        self.M = m
        self.eigen_value = None

    def set_attributes(self, dtr: np.ndarray, ltr: np.ndarray, dte: np.ndarray = None,  lte: np.ndarray = None):
        super().set_attributes(dtr, dte, ltr, lte)

    def process(self) -> tuple:
        # Evaluate the mean and center DTR and DTE over this mean
        mean = compute_mean(self.DTR)
        self.DTR = self.DTR - mean
        if self.DTE is not None:
            self.DTE = self.DTE - mean
        c = compute_covariance_matrix(self.DTR)
        # Evaluate the eigenvectors and directions
        self.eigen_value, u = np.linalg.eigh(c)
        p = u[:, ::-1][:, 0:self.M]
        # Compute the projections over the m directions p
        self.DTR = np.dot(p.T, self.DTR)
        if self.DTE is not None:
            self.DTE = np.dot(p.T, self.DTE)
        return self.DTR, self.DTE

    def __str__(self):
        return "PCA_M_" + str(self.M)


class Lda(Preprocess):

    def __init__(self, m: int):
        super().__init__()
        self.Sb = None
        self.Sw = None
        self.M = m

    def set_attributes(self, dtr: np.ndarray, ltr: np.ndarray, dte: np.ndarray = None, lte: np.ndarray = None):
        super().set_attributes(dtr, dte, ltr, lte)

    def __compute_Sb(self):
        sb = np.zeros((self.DTR.shape[0], self.DTR.shape[0]))
        mean_classes = [compute_mean(self.DTR[:, self.LTR == i]) for i in range(self.classes)]
        mean = compute_mean(self.DTR)
        self.DTR = self.DTR - mean
        for i in range(self.classes):
            sb += (self.LTR == i).sum() * (mean_classes[i] - mean).dot((mean_classes[i] - mean).T)
        self.Sb = (1 / self.DTR.shape[1]) * sb

    def __compute_Sw(self):
        mean_classes = [compute_mean(self.DTR[:, self.LTR == i]) for i in range(self.classes)]
        Sw = np.zeros((self.DTR.shape[0], self.DTR.shape[0]))
        for i in range(self.classes):
            Sw += (self.LTR == i).sum() * (1 / self.DTR[:, self.LTR == i].shape[1]) * \
                  (self.DTR[:, self.LTR == i] - mean_classes[i]).dot(
                      (self.DTR[:, self.LTR == i] - mean_classes[i]).T)
        self.Sw = (1 / self.DTR.shape[1]) * Sw

    def process(self) -> tuple:
        # Suppose that the data are already centered
        self.__compute_Sb()
        self.__compute_Sw()
        _, u = sp.linalg.eigh(self.Sb, self.Sw)
        p = u[:, ::-1][:, 0:self.M]
        self.DTR = np.dot(p.T, self.DTR)
        if self.DTE is not None:
            self.DTE = np.dot(p.T, self.DTE)
        return self.DTR, self.DTE

    def __str__(self):
        return "LDA_M_" + str(self.M)


class Zscore(Preprocess):

    def __init__(self):
        super().__init__()
        self.LTE = None
        self.LTR = None
        self.DTE = None
        self.DTR = None

    def process(self) -> tuple:
        mean = compute_mean(self.DTR)
        std = compute_std(self.DTR)
        self.DTR = (self.DTR - mean) / std
        if self.DTE is not None:
            self.DTE = (self.DTE - mean) / std
        return self.DTR, self.DTE

    def set_attributes(self, dtr: np.ndarray, ltr: np.ndarray, dte: np.ndarray = None, lte: np.ndarray = None):
        super().set_attributes(dtr, dte, ltr, lte)

    def __str__(self):
        return "ZSCORE"
