import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc


def fold_5(TPR, FPR, PR):
    fold = len(TPR)
    le = []
    for i in range(fold):
        le.append(len(TPR[i]))
    min_f = min(le)
    F_TPR = np.zeros((fold, min_f))
    F_FPR = np.zeros((fold, min_f))
    F_P = np.zeros((fold, min_f))
    for i in range(fold):
        k = len(TPR[i]) / min_f
        for j in range(min_f):
            F_TPR[i][j] = TPR[i][int(round(((j + 1) * k))) - 1]
            F_FPR[i][j] = FPR[i][int(round(((j + 1) * k))) - 1]
            F_P[i][j] = PR[i][int(round(((j + 1) * k))) - 1]
    TPR_5 = F_TPR.sum(0) / fold
    FPR_5 = F_FPR.sum(0) / fold
    PR_5 = F_P.sum(0) / fold
    return TPR_5, FPR_5, PR_5


def curve(FPR, TPR, P):
    plt.figure()
    plt.subplot(121)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title("ROC curve  (AUC = %.4f)" % (auc(FPR, TPR)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(FPR, TPR)

    plt.subplot(122)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title("PR curve  (AUPR = %.4f)" % (auc(TPR, P) + (TPR[0] * P[0])))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(TPR, P)
    plt.show()

    print("AUC:", auc(FPR, TPR), "AUPR:", auc(TPR, P) + (TPR[0] * P[0]))


def calculate_TPR_FPR(RD, f, B):
    old_id = np.argsort(-RD)
    min_f = int(min(f))
    max_f = int(max(f))
    TP_FN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    FP_TN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    TP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    TP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    FP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    FP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    P = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    P2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    for i in range(RD.shape[0]):
        TP_FN[i] = sum(B[i] == 1)
        FP_TN[i] = sum(B[i] == 0)
    for i in range(RD.shape[0]):
        for j in range(int(f[i])):
            if j == 0:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = 0
                    TP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = 0
                    FP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
            else:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = FP[i][j - 1]
                    TP[i][j] = TP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = TP[i][j - 1]
                    FP[i][j] = FP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
    ki = 0
    for i in range(RD.shape[0]):
        if TP_FN[i] == 0:
            TP[i] = 0
            FP[i] = 0
            ki = ki + 1
        else:
            TP[i] = TP[i] / TP_FN[i]
            FP[i] = FP[i] / FP_TN[i]
    for i in range(RD.shape[0]):
        kk = f[i] / min_f
        for j in range(min_f):
            TP2[i][j] = TP[i][int(np.round_(((j + 1) * kk))) - 1]
            FP2[i][j] = FP[i][int(np.round_(((j + 1) * kk))) - 1]
            P2[i][j] = P[i][int(np.round_(((j + 1) * kk))) - 1]
    TPR = TP2.sum(0) / (TP.shape[0] - ki)
    FPR = FP2.sum(0) / (FP.shape[0] - ki)
    P = P2.sum(0) / (P.shape[0] - ki)
    return TPR, FPR, P
