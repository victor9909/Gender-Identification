import numpy as np
import scipy.special
from project_lib.models.gaussian_models import *


def lbg_algorithm(iterations, X, start_gmm, alpha, psi, fun_mod=None):

    if fun_mod is not None:
        start_gmm = fun_mod(start_gmm, [X.shape[1]], X.shape[1])
    for i in range(len(start_gmm)):
        transformed_sigma = start_gmm[i][2]
        u, s, _ = np.linalg.svd(transformed_sigma)
        s[s < psi] = psi
        start_gmm[i] = (start_gmm[i][0], start_gmm[i]
                        [1], np.dot(u, v_col(s)*u.T))
    start_gmm = em_algorithm(X, start_gmm, psi, fun_mod)

    for i in range(iterations):
        gmm_new = list()
        for g in start_gmm:
            sigma_g = g[2]
            u, s, _ = np.linalg.svd(sigma_g)
            d = u[:, 0:1] * s[0]**0.5 * alpha
            new_w = g[0]/2
            gmm_new.append((new_w, g[1] + d, sigma_g))
            gmm_new.append((new_w, g[1] - d, sigma_g))
        start_gmm = em_algorithm(X, gmm_new, psi, fun_mod)
    return start_gmm


def em_algorithm(X, gmm, psi, fun_mod=None):
    ll_new = None
    ll_old = None
    while ll_old is None or ll_new - ll_old > 1e-6:
        ll_old = ll_new
        s_joint = np.zeros((len(gmm), X.shape[1]))
        for g in range(len(gmm)):
            s_joint[g, :] = logpdf_GAU_ND(
                X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        s_marginal = scipy.special.logsumexp(s_joint, axis=0)
        ll_new = s_marginal.sum() / X.shape[1]
        P = np.exp(s_joint - s_marginal)
        gmm_new = []
        z_vec = np.zeros(len(gmm))
        for g in range(len(gmm)):
            gamma = P[g, :]
            zero_order = gamma.sum()
            z_vec[g] = zero_order
            first_order = (v_row(gamma) * X).sum(1)
            second_order = np.dot(X, (v_row(gamma) * X).T)
            w = zero_order / X.shape[1]
            mu = v_col(first_order / zero_order)
            sigma = second_order / zero_order - np.dot(mu, mu.T)
            gmm_new.append((w, mu, sigma))

        if fun_mod is not None:
            gmm_new = fun_mod(gmm_new, z_vec, X.shape[1])

        for i in range(len(gmm)):
            transformed_sigma = gmm_new[i][2]
            u, s, _ = np.linalg.svd(transformed_sigma)
            s[s < psi] = psi
            gmm_new[i] = (gmm_new[i][0], gmm_new[i][1], np.dot(u, v_col(s) * u.T))
        gmm = gmm_new
    return gmm


def compute_gmm_scores(D, L, gmm):
    scores = np.zeros((np.unique(L).size, D.shape[1]))
    for classes in range(np.unique(L).size):
        scores[classes, :] = np.exp(logpdf_gmm(D, gmm[classes]))
    llr = np.zeros(scores.shape[1])
    for i in range(scores.shape[1]):
        llr[i] = np.log(scores[1, i] / scores[0, i])
    return llr


def logpdf_gmm(X, gmm):
    s = np.zeros((len(gmm), X.shape[1]))
    for i in range(X.shape[1]):
        for (idx, component) in enumerate(gmm):
            s[idx, i] = logpdf_GAU_ND(X[:, i:i+1], component[1], component[2]) + np.log(component[0])
    return scipy.special.logsumexp(s, axis=0)


class GMM(Models):

    def __init__(self, iterations, alpha=0.1, psi=0.01):
        super().__init__([])
        self.gmm = None
        self.params = [('components', 2**iterations), ('alpha', alpha), ('psi', psi)]
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi

    def compute_scores(self):
        self.scores = compute_gmm_scores(self.DTE,  self.LTE, self.gmm)

    def train(self):
        gmm = list()
        for classes in range(np.unique(self.LTR).size):
            mu = compute_mean(self.DTR[:, self.LTR == classes])
            cov = compute_covariance_matrix(self.DTR[:, self.LTR == classes])
            gmm.append(lbg_algorithm(self.iterations, self.DTR[:, self.LTR == classes], [[1, mu, cov]], 0.1, 0.01))
        self.gmm = gmm

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        super().set_attributes(DTR, LTR, DTE, LTE)

    def __str__(self):
        return "GMM_"


class GMMTied(Models):

    def __init__(self, iterations, alpha=0.1, psi=0.01):
        super().__init__([])
        self.gmm = None
        self.params = [('components', 2**iterations), ('alpha', alpha), ('psi', psi)]
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi

    def compute_scores(self):
        self.scores = compute_gmm_scores(self.DTE, self.LTE, self.gmm)

    @staticmethod
    def gmm_tied_function(gmm, z_vec, n):
        tied_sigma = np.zeros((gmm[0][2].shape[0], gmm[0][2].shape[0]))
        for g in range((len(gmm))):
            tied_sigma += gmm[g][2] * z_vec[g]
        tied_sigma = (1 / n) * tied_sigma
        for g in range((len(gmm))):
            gmm[g] = (gmm[g][0], gmm[g][1], tied_sigma)
        return gmm

    def train(self):
        gmm = list()
        for classes in range(np.unique(self.LTR).size):
            mu = compute_mean(self.DTR[:, self.LTR == classes])
            cov = compute_covariance_matrix(self.DTR[:, self.LTR == classes])
            gmm.append(lbg_algorithm(self.iterations, self.DTR[:, self.LTR == classes], [[1, mu, cov]], 0.1, 0.01,
                                     self.gmm_tied_function))
        self.gmm = gmm

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        super().set_attributes(DTR, LTR, DTE, LTE)

    def __str__(self):
        return "GMMTied_"


class GMMDiagonal(Models):

    def __init__(self, iterations, alpha=0.1, psi=0.01):
        super().__init__([])
        self.params = [('components', 2**iterations), ('alpha', alpha), ('psi', psi)]
        self.gmm = None
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi

    def compute_scores(self):
        self.scores = compute_gmm_scores(self.DTE, self.LTE, self.gmm)

    @staticmethod
    def gmm_diagonal_function(gmm, _z_vec, _n):
        for g in range((len(gmm))):
            sigma = gmm[g][2] * np.eye(gmm[g][2].shape[0])
            gmm[g] = (gmm[g][0], gmm[g][1], sigma)
        return gmm

    def train(self):
        gmm = list()
        for classes in range(np.unique(self.LTR).size):
            mu = compute_mean(self.DTR[:, self.LTR == classes])
            cov = compute_covariance_matrix(self.DTR[:, self.LTR == classes])
            gmm.append(lbg_algorithm(self.iterations, self.DTR[:, self.LTR == classes], [[1, mu, cov]], 0.1, 0.01,
                                     self.gmm_diagonal_function))
        self.gmm = gmm

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        super().set_attributes(DTR, LTR, DTE, LTE)

    def __str__(self):
        return "GMMDiagonal_"


class GMMTiedDiagonal(Models):

    def __init__(self, iterations, alpha=0.1, psi=0.01):
        super().__init__([])
        self.params = [('components', 2**iterations), ('alpha', alpha), ('psi', psi)]
        self.gmm = None
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi

    def compute_scores(self):
        self.scores = compute_gmm_scores(self.DTE, self.LTE, self.gmm)

    @staticmethod
    def __gmm_tied_diagonal_function(gmm, z_vec, n):
        tied_gmm = GMMTied.gmm_tied_function(gmm, z_vec, n)
        tied_diagonal_gmm = GMMDiagonal.gmm_diagonal_function(tied_gmm, z_vec, n)
        return tied_diagonal_gmm

    def train(self):
        gmm = list()
        for classes in range(np.unique(self.LTR).size):
            mu = compute_mean(self.DTR[:, self.LTR == classes])
            cov = compute_covariance_matrix(self.DTR[:, self.LTR == classes])
            gmm.append(lbg_algorithm(self.iterations, self.DTR[:, self.LTR == classes], [[1, mu, cov]], 0.1, 0.01,
                                     self.__gmm_tied_diagonal_function))
        self.gmm = gmm

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        super().set_attributes(DTR, LTR, DTE, LTE)

    def __str__(self):
        return "GMMTiedDiagonal_"
