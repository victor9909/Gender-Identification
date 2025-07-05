from project_lib.models.gaussian_models import *
from project_lib.models.pipe import *
from project_lib.preprocessing import *


def evaluation(DTR, LTR, DTE, LTE):
    print("Starting testing Gaussian Classifiers...")
    print("========================================================")
    print("Training TMVG with ZScored features...")
    pipeTMVG = Pipe(DTR, LTR, DTE, LTE, [Zscore()], TMVG())
    pipeTMVG.make_evaluation()
    print("End training TMVG with ZScored features...")
    print("========================================================")
    print("Finish testing on gaussian classifiers...")