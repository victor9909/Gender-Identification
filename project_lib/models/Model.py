from abc import abstractmethod
import numpy as np
import enum

class Models:

    def __init__(self, priors: list):
        self.priors = priors
        self.scores = None
        self.DTR = None
        self.LTR = None
        self.DTE = None
        self.LTE = None

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def compute_scores(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Model(enum.Enum):
    MVG = 0
    TNB = 1
    NBG = 2
    TMVG = 3
    LR = 4
    QLR = 5
    SVM = 6
    PolSVM = 7
    RadialBasedSVM = 8
    GMMTied = 9
    GMMDiagonal = 10
    GMMTiedDiagonal = 11
    GMM = 12
    Fusion = 13

