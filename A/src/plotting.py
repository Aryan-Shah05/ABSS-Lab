import os
import numpy as np
import matplotlib.pyplot as plt


def plot_distributions(genuine, imposter, outdir, metric_name):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.hist(imposter, bins=100, alpha=0.6, label='Imposter')
    plt.hist(genuine, bins=100, alpha=0.6, label='Genuine')
    plt.legend()
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title(f'Genuine vs Imposter Distributions ({metric_name})')
    path = os.path.join(outdir, f'distributions_{metric_name}.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_far_frr(thresholds, far, frr, outdir, metric_name):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(thresholds, far, label='FAR')
    plt.plot(thresholds, frr, label='FRR')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title(f'FAR and FRR vs Threshold ({metric_name})')
    plt.legend()
    path = os.path.join(outdir, f'far_frr_{metric_name}.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_roc(far, frr, outdir, metric_name):
    # TPR = 1 - FRR, FPR = FAR
    tpr = 1 - frr
    fpr = far
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({metric_name})')
    plt.tight_layout()
    path = os.path.join(outdir, f'roc_{metric_name}.png')
    plt.savefig(path)
    plt.close()
    return path
