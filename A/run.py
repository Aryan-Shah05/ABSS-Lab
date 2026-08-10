#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from src import biometrics, plotting


def ensure_out(out):
    os.makedirs(out, exist_ok=True)


def analyze(data_path, outdir):
    users = biometrics.load_data(data_path)
    templates = biometrics.make_templates(users)

    results = {}
    for metric in ["euclidean", "cosine"]:
        genuine, imposter, all_scores = biometrics.compute_scores(users, templates, metric=metric)
        thresholds = biometrics.thresholds_from_scores(genuine, imposter, n_steps=1000)
        far, frr = biometrics.compute_far_frr(genuine, imposter, thresholds)
        eer, thr = biometrics.compute_eer(thresholds, far, frr)
        dprime = biometrics.decidability_index(genuine, imposter)

        mdir = os.path.join(outdir, metric)
        os.makedirs(mdir, exist_ok=True)
        dist_path = plotting.plot_distributions(genuine, imposter, mdir, metric)
        ff_path = plotting.plot_far_frr(thresholds, far, frr, mdir, metric)
        roc_path = plotting.plot_roc(far, frr, mdir, metric)

        results[metric] = {
            "eer": float(eer),
            "eer_threshold": float(thr),
            "decidability": float(dprime),
            "plots": {
                "distributions": dist_path,
                "far_frr": ff_path,
                "roc": roc_path,
            },
        }

        # save raw scores
        np.savetxt(os.path.join(mdir, 'genuine_scores.txt'), genuine)
        np.savetxt(os.path.join(mdir, 'imposter_scores.txt'), imposter)

    # write summary
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('Analysis complete. Results in', outdir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to biomet_data.csv')
    p.add_argument('--out', default='results', help='Output folder')
    args = p.parse_args()
    ensure_out(args.out)
    analyze(args.data, args.out)


if __name__ == '__main__':
    main()
