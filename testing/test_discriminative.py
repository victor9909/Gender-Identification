from project_lib.models.discriminative_models import *
from project_lib.models.pipe import *
from project_lib.graphics import compute_std_minCDF_plot_for_model
from project_lib.preprocessing import *


def evaluation(DTR, LTR, DTE, LTE):

    print("Testing discriminative models...")
    l_val = np.logspace(-5, 5, num=51)
    print("================== Start LR Testing ===================")
    for x in l_val:
        for priors in [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]:
            print("========================================================")
            print("Testing " + str(LR([], None)).replace("_", "") +
                  "(πT=" + str(priors[0]) + "; λ=" + str(x) + ") with ZScore features...")
            pipeLR = Pipe(DTR, LTR, DTE, LTE, [Zscore()], LR(priors, x))
            pipeLR.make_evaluation()
        print("========================================================")
    print("=================== End LR Testing ====================")

    print("================== Start QLR Testing ==================")
    for x in l_val:
        for priors in [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]:
            print("Testing " + str(QLR([], None)).replace("_", "") +
                  "(πT=" + str(priors[0]) + "; λ=" + str(x) + ") with RAW features...")
            pipeQLR = Pipe(DTR, LTR, DTE, LTE, [], QLR(priors, x))
            pipeQLR.make_evaluation()
    print("================== End QLR Testing ====================")

    print("End Testing discriminative models...")

    compute_std_minCDF_plot_for_model(Model.LR, ["ZScore"], {"prior_t": 0.9}, is_train=False)


