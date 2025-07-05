from project_lib.models.GMM_models import *
from project_lib.models.pipe import *
from project_lib.graphics import compute_std_minCDF_plot_for_model
from project_lib.preprocessing import *


def eval_single_gmm(DTR, LTR, DTE, LTE, iterations):
    print("=======================ZSCORE========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore()], GMM(iterations))
    pipe_gmm.make_evaluation()
    print("========================RAW============================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [], GMM(iterations))
    pipe_gmm.make_evaluation()


def eval_gmm(DTR, LTR, DTE, LTE):
    print("========================================================")
    for iteration in range(7):
        print("========================================================")
        print("========================================================")
        print("Testing for GMM Model with " + str(2 ** iteration) + " components for all pre process in report...")
        eval_single_gmm(DTR, LTR, DTE, LTE, iteration)
        print("========================================================")
        print("========================================================")
    print("========================================================")


def eval_single_gmm_tied(DTR, LTR, DTE, LTE, iterations):
    print("=======================ZSCORE========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore()], GMMTied(iterations))
    pipe_gmm.make_evaluation()
    print("========================RAW============================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [], GMMTied(iterations))
    pipe_gmm.make_evaluation()


def eval_gmm_tied(DTR, LTR, DTE, LTE):
    print("========================================================")
    for iteration in range(7):
        print("========================================================")
        print("========================================================")
        print("Testing for GMMTied Model with " + str(2 ** iteration) + " components for all pre process in report...")
        eval_single_gmm_tied(DTR, LTR, DTE, LTE, iteration)
        print("========================================================")
        print("========================================================")
    print("========================================================")


def eval_single_gmm_diagonal(DTR, LTR, DTE, LTE, iterations):
    print("===================ZSCORE+PCA(12)=======================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore(), Pca(12)], GMMDiagonal(iterations))
    pipe_gmm.make_evaluation()
    print("=====================RAW+PCA(12)========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Pca(12)], GMMDiagonal(iterations))
    pipe_gmm.make_evaluation()


def eval_gmm_diagonal(DTR, LTR, DTE, LTE):
    print("========================================================")
    for iteration in range(7):
        print("========================================================")
        print("========================================================")
        print("Testing for GMMDiagonal Model with " + str(
            2 ** iteration) + " components for all pre process in report...")
        eval_single_gmm_diagonal(DTR, LTR, DTE, LTE, iteration)
        print("========================================================")
        print("========================================================")
    print("========================================================")


def eval_single_gmm_tied_diagonal(DTR, LTR, DTE, LTE, iterations):
    print("===================ZSCORE+PCA(12)=======================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Zscore(), Pca(12)], GMMTiedDiagonal(iterations))
    pipe_gmm.make_evaluation()

    print("=====================RAW+PCA(12)========================")
    pipe_gmm = Pipe(DTR, LTR, DTE, LTE, [Pca(12)], GMMTiedDiagonal(iterations))
    pipe_gmm.make_evaluation()


def eval_gmm_tied_diagonal(DTR, LTR, DTE, LTE):
    print("========================================================")
    for iteration in range(7):
        print("========================================================")
        print("========================================================")
        print("Testing for GMMTiedDiagonal Model with " + str(
            2 ** iteration) + " components for all pre process in report...")
        eval_single_gmm_tied_diagonal(DTR, LTR, DTE, LTE, iteration)
        print("========================================================")
        print("========================================================")
    print("========================================================")


def evaluation(DTR, LTR, DTE, LTE):
    eval_gmm(DTR, LTR, DTE, LTE)
    eval_gmm_tied(DTR, LTR, DTE, LTE)
    eval_gmm_diagonal(DTR, LTR, DTE, LTE)
    eval_gmm_tied_diagonal(DTR, LTR, DTE, LTE)

    for gmm_model in [Model.GMMTiedDiagonal, Model.GMMDiagonal]:
        compute_std_minCDF_plot_for_model(gmm_model, [], {}, param_plot=["ZScore_pca_m_12", "pca_m_12"], is_train=False)

    for gmm_model in [Model.GMM, Model.GMMTied]:
        compute_std_minCDF_plot_for_model(gmm_model, [], {}, param_plot=["ZScore", "Raw"], is_train=False)

