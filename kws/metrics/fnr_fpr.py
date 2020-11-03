import numpy as np
from sklearn.metrics import auc


def fnr_fpr_curve(probs, targets):
    # calculates FNR(t)/FPR(t) curve 
    # for all possible thresholds t

    order = np.argsort(probs)
    probs = probs[order]
    targets = targets[order]

    pos_examples = (targets == 1).sum()
    fnr = np.cumsum(targets == 1) / pos_examples
    fnr = np.concatenate([[0.0], fnr])

    neg_examples = (targets == 0).sum()
    fpr = 1.0 - np.cumsum(targets == 0) / neg_examples
    fpr = np.concatenate([[1.0], fpr])

    return fnr, fpr


def fnr_fpr_auc(probs, targets):
    # calucaltes area under FNR/FPR curve

    fnr, fpr = fnr_fpr_curve(probs, targets)
    return auc(fpr, fnr)


def fnr_at_fpr(probs, targets, max_fpr=0.1):
    # calculates FNR for specific FPR

    fnr, fpr = fnr_fpr_curve(probs, targets)
    index = np.sum(fpr > max_fpr) - 1
    return fnr[index]


def fr_at_fa(probs, targets, max_fa_per_hour=1.0, audio_seconds=1.0):
    # calculates falce rejects rate for specific 
    # false alarms per hour

    max_fpr = audio_seconds / (3600 * max_fa_per_hour)
    return fnr_at_fpr(probs, targets, max_fpr)
