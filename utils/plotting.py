import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plotROC_index(fpr, tpr, roc_auc, index):
    if( index >= len(fpr)):
        print("Wrong index "+index)
    plt.figure()
    lw = 2
    plt.plot(fpr[index], tpr[index], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[index])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plotROC(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


# def plotROCs(fpr, tpr, roc_auc, items, plot_user_auc = False):
#     lw = 2
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#     for i in fpr:
#         if( plot_user_auc ):
#             plt.plot(fpr[i], tpr[i], lw=lw, alpha=.3, label='user %d (AUC = %0.4f)' % (i, roc_auc[i]) )
#         tprs.append(np.interp(mean_fpr, fpr[i], tpr[i]))
#         tprs[-1][0] = 0.0
#         aucs.append(roc_auc[i])
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

#     # plot mean
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     plt.plot(mean_fpr, mean_tpr, color='b',
#              label=r'Mean (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
#              lw=2, alpha=.8)

#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                      label=r'$\pm$ 1 std. dev.')

#     # end plot mean
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     titlestr= "ROC CURVE"
#     plt.title(titlestr)
#     plt.legend(loc="lower right")
#     plt.show()


# def plot_precisions_recalls(precisions, recalls, pr_aucs):
#     precisions_interp = []
#     aucs = []
#     mean_recalls = np.linspace(0, 1, 100)

#     for i in precisions:
#         precisions_interp.append(np.interp(mean_recalls, np.flip(recalls[i]), np.flip(precisions[i])))
#         aucs.append(pr_aucs[i])

#     # plot mean
#     mean_precision = np.mean(precisions_interp, axis=0)
#     mean_auc = auc(mean_recalls, mean_precision)
#     std_auc = np.std(aucs)

#     plt.plot(mean_recalls, mean_precision, color='b',
#              label=r'Mean (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
#              lw=2, alpha=.8)

#     std_precision = np.std(precisions_interp, axis=0)
#     precisions_upper = np.minimum(mean_precision + std_precision, 1)
#     precisions_lower = np.maximum(mean_precision - std_precision, 0)
    
#     plt.fill_between(mean_recalls, precisions_lower, precisions_upper, color='grey', alpha=.2,
#                      label=r'$\pm$ 1 std. dev.')

#     # end plot mean
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     title_str = "Precision-Recall Curve"
#     plt.title(title_str)
#     plt.legend(loc="upper right")
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

def plotROCs(fpr, tpr, roc_auc, items, plot_user_auc=False, save_to_file=False, filename="roc_curve.png"):
    lw = 2
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for i in fpr:
        if plot_user_auc:
            plt.plot(fpr[i], tpr[i], lw=lw, alpha=.3, label='user %d (AUC = %0.4f)' % (i, roc_auc[i]))

        tprs.append(np.interp(mean_fpr, fpr[i], tpr[i]))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc[i])
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    # plot mean
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    # end plot mean
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title_str = "ROC CURVE"
    plt.title(title_str)
    plt.legend(loc="lower right")

    if save_to_file:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()



def plot_precisions_recalls(precisions, recalls, pr_aucs, save_to_file=False, filename="precision_recall_curve.png"):
    precisions_interp = []
    aucs = []
    mean_recalls = np.linspace(0, 1, 100)

    for i in precisions:
        precisions_interp.append(np.interp(mean_recalls, np.flip(recalls[i]), np.flip(precisions[i])))
        aucs.append(pr_aucs[i])

    # plot mean
    mean_precision = np.mean(precisions_interp, axis=0)
    mean_auc = auc(mean_recalls, mean_precision)
    std_auc = np.std(aucs)

    plt.plot(mean_recalls, mean_precision, color='b',
             label=r'Mean (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_precision = np.std(precisions_interp, axis=0)
    precisions_upper = np.minimum(mean_precision + std_precision, 1)
    precisions_lower = np.maximum(mean_precision - std_precision, 0)
    
    plt.fill_between(mean_recalls, precisions_lower, precisions_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    # end plot mean
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    title_str = "Precision-Recall Curve"
    plt.title(title_str)
    plt.legend(loc="upper right")

    if save_to_file:
        plt.savefig(filename)
    else:
        plt.show()
    
    plt.close()
