import os
import numpy as np
from scipy.spatial.distance import cdist


def load_data(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        # maybe single row line-wrapped; try reading by splitting whitespace
        arr = np.genfromtxt(path)
    if arr.size != 1000 * 144:
        # try to infer rows: if there are 144 columns, good
        if arr.shape[1] != 144:
            raise ValueError(f"Unexpected data shape {arr.shape}, expected total 1000x144")
    # ensure shape (1000,144)
    data = arr.reshape(-1, 144)
    if data.shape[0] != 1000:
        raise ValueError(f"Expected 1000 samples, got {data.shape[0]}")
    # reshape into users x samples x features
    users = data.reshape(100, 10, 144)
    return users


def make_templates(users):
    # users shape: (100,10,144). enrollment samples are first 5
    enroll = users[:, :5, :]
    templates = enroll.mean(axis=1)
    return templates


def compute_scores(users, templates, metric="euclidean"):
    # iterate test samples
    n_users = users.shape[0]
    genuine = []
    imposter = []
    all_scores = []
    for u in range(n_users):
        tests = users[u, 5:10, :]
        for t in tests:
            # compute distances/similarities between t and all templates
            if metric == "euclidean":
                d = np.linalg.norm(templates - t, axis=1)
                # convert to similarity-like score (higher better)
                scores = -d
            elif metric == "cosine":
                # cosine similarity
                # normalized dot
                t_norm = t / np.linalg.norm(t)
                temp_norm = templates / np.linalg.norm(templates, axis=1, keepdims=True)
                scores = (temp_norm @ t_norm)
            else:
                # use scipy cdist for other metrics
                d = cdist(templates, t.reshape(1, -1), metric=metric).ravel()
                scores = -d

            all_scores.append(scores)
            # genuine score is score with own template (index u)
            genuine.append(scores[u])
            # imposter scores are scores with templates of other users
            imposter.extend(np.delete(scores, u))

    genuine = np.array(genuine)
    imposter = np.array(imposter)
    all_scores = np.array(all_scores)  # shape (500,100)
    return genuine, imposter, all_scores


def thresholds_from_scores(genuine, imposter, n_steps=1000):
    lo = min(genuine.min(), imposter.min())
    hi = max(genuine.max(), imposter.max())
    return np.linspace(lo, hi, n_steps)


def compute_far_frr(genuine, imposter, thresholds):
    genu = genuine[:, None]
    impost = imposter[:, None]
    # For threshold, accept if score >= threshold
    far = []
    frr = []
    for thr in thresholds:
        fa = np.mean(impost >= thr)
        fr = np.mean(genu < thr)
        far.append(fa)
        frr.append(fr)
    return np.array(far), np.array(frr)


def compute_eer(thresholds, far, frr):
    # EER is where FAR and FRR are closest
    idx = np.argmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2.0
    return eer, thresholds[idx]


def decidability_index(genuine, imposter):
    mu_g = np.mean(genuine)
    mu_i = np.mean(imposter)
    sd_g = np.std(genuine)
    sd_i = np.std(imposter)
    dprime = abs(mu_g - mu_i) / np.sqrt(0.5 * (sd_g ** 2 + sd_i ** 2))
    return dprime
