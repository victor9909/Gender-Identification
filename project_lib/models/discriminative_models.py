import numpy as np

from project_lib.models.Model import *
from project_lib.utils import *
import matplotlib.pyplot as plt
from project_lib.preprocessing import *


class LR(Models):

    def __init__(self, priors, l):
        super().__init__(priors)
        self.l = l
        self.params = [('l', l)]
        self.w = None
        self.b = None

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        super().set_attributes(DTR, LTR, DTE, LTE)

    def train(self):
        fun_to_minimize = logreg_obj_wrap(self)
        minim, j, _ = sp.optimize.fmin_l_bfgs_b(fun_to_minimize, np.zeros((self.DTR.shape[0] + 1)), approx_grad=True,
                                                factr=1.0)
        self.w = minim[0:-1]
        self.b = minim[-1]

    def compute_scores(self):
        self.scores = np.dot(self.w.T, self.DTE) + self.b

    def __str__(self):
        return "LR_"


class QLR(LR):

    def __init__(self, priors, l):
        super().__init__(priors, l)

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        super().set_attributes(DTR, LTR, DTE, LTE)

    def train(self):
        (self.DTR, self.DTE) = polynomial_transformation(self.DTR, self.DTE)
        super().train()

    def compute_scores(self):
        super().compute_scores()

    def __str__(self):
        return "QLR_"


def logreg_obj_wrap(model: LR):
    z = 2 * model.LTR - 1

    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        s = 0
        const = (model.l / 2) * (np.dot(w, w.T))
        for i in range(np.unique(model.LTR).size):
            const_class = (model.priors[i] / model.LTR[model.LTR == i].size)
            s += const_class * np.logaddexp(0, -z[model.LTR == i] * (np.dot(w.T, model.DTR[:, model.LTR == i]) + b)).sum()

        return const + s

    return logreg_obj


def compute_LR_score_matrix(D: np.ndarray, W: np.ndarray, b: np.ndarray):
    return np.dot(W.T, D) + b


def polynomial_transformation(DTR, DTE):
    n_train = DTR.shape[1]
    n_eval = DTE.shape[1]
    n_f = DTR.shape[0] ** 2 + DTR.shape[0]
    quad_dtr = np.zeros((n_f, n_train))
    quad_dte = np.zeros((n_f, n_eval))

    for i in range(n_train):
        quad_dtr[:, i:i + 1] = stack(DTR[:, i:i + 1])
    for i in range(n_eval):
        quad_dte[:, i:i + 1] = stack(DTE[:, i:i + 1])

    return quad_dtr, quad_dte


def stack(array):
    n_f = array.shape[0]
    xx_t = np.dot(array, array.T)
    column = np.zeros((n_f ** 2 + n_f, 1))
    for i in range(n_f):
        column[i * n_f:i * n_f + n_f, :] = xx_t[:, i:i + 1]
    column[n_f ** 2: n_f ** 2 + n_f, :] = array
    return column
