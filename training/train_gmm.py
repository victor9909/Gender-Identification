from project_lib.models.GMM_models import *
from project_lib.models.pipe import *
from project_lib.graphics import compute_std_minCDF_plot_for_model
from project_lib.preprocessing import *


def train_single_gmm(DTR, LTR, DTE, LTE, iterations):

    print("=======================ZSCORE========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore()], GMM(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("========================RAW============================")

    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [], GMM(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("=====================RAW+PCA(12)========================")

    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Pca(12)], GMM(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("===================ZSCORE+PCA(12)=======================")

    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore(), Pca(12)], GMM(iterations))
    pipe_gmm.make_train_with_K_fold()


def train_gmm(DTR, LTR, DTE, LTE):

    print("========================================================")
    for iteration in range(7):
        print("========================================================")
        print("========================================================")
        print("Training for GMM Model with " + str(2**iteration) + " components for all pre process in report...")
        train_single_gmm(DTR, LTR, DTE, LTE, iteration)
        print("========================================================")
        print("========================================================")
    print("========================================================")


def train_single_gmm_tied(DTR, LTR, DTE, LTE, iterations):
    print("=======================ZSCORE========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore()], GMMTied(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("========================RAW============================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [], GMMTied(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("=====================RAW+PCA(12)========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Pca(12)], GMMTied(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("===================ZSCORE+PCA(12)=======================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore(), Pca(12)], GMMTied(iterations))
    pipe_gmm.make_train_with_K_fold()


def train_gmm_tied(DTR, LTR, DTE, LTE):
    print("========================================================")
    for iteration in range(7):
        print("========================================================")
        print("========================================================")
        print("Training for GMMTied Model with " + str(2 ** iteration) + " components for all pre process in report...")
        train_single_gmm_tied(DTR, LTR, DTE, LTE, iteration)
        print("========================================================")
        print("========================================================")
    print("========================================================")


def train_single_gmm_diagonal(DTR, LTR, DTE, LTE, iterations):
    print("=======================ZSCORE========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore()], GMMDiagonal(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("========================RAW============================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [], GMMDiagonal(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("=====================RAW+PCA(12)========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Pca(12)], GMMDiagonal(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("===================ZSCORE+PCA(12)=======================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore(), Pca(12)], GMMDiagonal(iterations))
    pipe_gmm.make_train_with_K_fold()


def train_gmm_diagonal(DTR, LTR, DTE, LTE):
    print("========================================================")
    for iteration in range(7):
        print("========================================================")
        print("========================================================")
        print("Training for GMMDiagonal Model with " + str(2 ** iteration) + " components for all pre process in report...")
        train_single_gmm_diagonal(DTR, LTR, DTE, LTE, iteration)
        print("========================================================")
        print("========================================================")
    print("========================================================")


def train_single_gmm_tied_diagonal(DTR, LTR, DTE, LTE, iterations):
    print("=======================ZSCORE========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore()], GMMTiedDiagonal(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("========================RAW============================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [], GMMTiedDiagonal(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("=====================RAW+PCA(12)========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Pca(12)], GMMTiedDiagonal(iterations))
    pipe_gmm.make_train_with_K_fold()
    print("===================ZSCORE+PCA(12)=======================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore(), Pca(12)], GMMTiedDiagonal(iterations))
    pipe_gmm.make_train_with_K_fold()


def train_gmm_tied_diagonal(DTR, LTR, DTE, LTE):
    print("========================================================")
    for iteration in range(7):
        print("========================================================")
        print("========================================================")
        print("Training for GMMTiedDiagonal Model with " + str(
            2 ** iteration) + " components for all pre process in report...")
        train_single_gmm_tied_diagonal(DTR, LTR, DTE, LTE, iteration)
        print("========================================================")
        print("========================================================")
    print("========================================================")


def train(DTR, LTR, DTE, LTE):

    train_gmm(DTR, LTR, DTE, LTE)
    train_gmm_tied(DTR, LTR, DTE, LTE)
    train_gmm_diagonal(DTR, LTR, DTE, LTE)
    train_gmm_tied_diagonal(DTR, LTR, DTE, LTE)

    for gmm_model in [Model.GMM, Model.GMMTied, Model.GMMTiedDiagonal, Model.GMMDiagonal]:
        compute_std_minCDF_plot_for_model(gmm_model, [], {}, param_plot=["ZScore", "Raw"])
        compute_std_minCDF_plot_for_model(gmm_model, [], {}, param_plot=["ZScore_pca_m_12", "pca_m_12"])

