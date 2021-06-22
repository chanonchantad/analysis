from sklearn.metrics import roc_curve, auc
from sklearn import metrics

import matplotlib.pyplot as plt

# --------------------------------------------------
# This file contains code to create a ROC/AUC analysis of 
# an AI trained model to visualize how accurate the model
# is in a graph form across all possible thresholds.
# --------------------------------------------------

def roc(gts, sfts, gt_lbl=1, title='My Graph'):
    
    # --- calculate the auc (FPR = false positive rate, TPR = true positive rate)
    fpr, tpr, thresholds = metrics.roc_curve(gts, sft, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    
    # --- draw out graph using matplotlib
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.001, 1.0])
    plt.ylim([-0.001, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc