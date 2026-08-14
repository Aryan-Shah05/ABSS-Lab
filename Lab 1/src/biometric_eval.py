from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "biomet_data.csv"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

N_USERS = 100
SAMPLES_PER_USER = 10
TRAIN_PER_USER = 5
TEST_PER_USER = 5
N_FEATURES = 144


@dataclass(frozen=True)
class Evaluation:
    name: str
    genuine: np.ndarray
    impostor: np.ndarray
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray
    tpr: np.ndarray
    fpr: np.ndarray
    eer: float
    eer_threshold: float
    decidability_index: float
    lower_is_match: bool


def load_features(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=float)
    if raw.shape == (N_FEATURES, N_USERS * SAMPLES_PER_USER):
        raw = raw.T
    elif raw.shape != (N_USERS * SAMPLES_PER_USER, N_FEATURES):
        raise ValueError(f"Expected 144x1000 or 1000x144 matrix, got {raw.shape}")
    return raw.reshape(N_USERS, SAMPLES_PER_USER, N_FEATURES)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True).T
    denom = np.maximum(a_norm @ b_norm, np.finfo(float).eps)
    return (a @ b.T) / denom


def score_samples(features: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    templates = features[:, :TRAIN_PER_USER, :].mean(axis=1)
    tests = features[:, TRAIN_PER_USER:, :].reshape(N_USERS * TEST_PER_USER, N_FEATURES)
    labels = np.repeat(np.arange(N_USERS), TEST_PER_USER)

    if metric == "euclidean":
        scores = np.linalg.norm(tests[:, None, :] - templates[None, :, :], axis=2)
        lower_is_match = True
    elif metric == "cosine":
        scores = cosine_similarity_matrix(tests, templates)
        lower_is_match = False
    else:
        raise ValueError(metric)

    genuine = scores[np.arange(len(tests)), labels]
    impostor = scores[labels[:, None] != np.arange(N_USERS)].reshape(len(tests), N_USERS - 1).ravel()
    return genuine, impostor, lower_is_match


def compute_curves(name: str, genuine: np.ndarray, impostor: np.ndarray, lower_is_match: bool) -> Evaluation:
    all_scores = np.concatenate([genuine, impostor])
    margin = (all_scores.max() - all_scores.min()) * 0.01 or 1.0
    thresholds = np.linspace(all_scores.min() - margin, all_scores.max() + margin, 2000)

    if lower_is_match:
        genuine_accepts = genuine[:, None] <= thresholds[None, :]
        impostor_accepts = impostor[:, None] <= thresholds[None, :]
    else:
        genuine_accepts = genuine[:, None] >= thresholds[None, :]
        impostor_accepts = impostor[:, None] >= thresholds[None, :]

    frr = 1.0 - genuine_accepts.mean(axis=0)
    far = impostor_accepts.mean(axis=0)
    tpr = 1.0 - frr
    fpr = far
    idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[idx] + frr[idx]) / 2.0)

    mean_g, mean_i = genuine.mean(), impostor.mean()
    var_g, var_i = genuine.var(ddof=1), impostor.var(ddof=1)
    decidability = float(abs(mean_g - mean_i) / math.sqrt(0.5 * (var_g + var_i)))

    return Evaluation(
        name=name,
        genuine=genuine,
        impostor=impostor,
        thresholds=thresholds,
        far=far,
        frr=frr,
        tpr=tpr,
        fpr=fpr,
        eer=eer,
        eer_threshold=float(thresholds[idx]),
        decidability_index=decidability,
        lower_is_match=lower_is_match,
    )


def svg_line_plot(path: Path, title: str, series: list[tuple[str, np.ndarray, np.ndarray, str]], x_label: str, y_label: str, force_unit_y: bool = False) -> None:
    w, h = 900, 560
    left, right, top, bottom = 80, 30, 55, 75
    xs = np.concatenate([s[1] for s in series])
    ys = np.concatenate([s[2] for s in series])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    if x_max == x_min: x_max += 1
    if y_max == y_min: y_max += 1
    y_min = min(0.0, y_min)
    if force_unit_y:
        y_max = max(1.0, y_max)
    else:
        y_max = y_max * 1.08

    def px(x): return left + (x - x_min) / (x_max - x_min) * (w - left - right)
    def py(y): return top + (y_max - y) / (y_max - y_min) * (h - top - bottom)

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>']
    lines.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{title}</text>')
    lines.append(f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#333"/>')
    for t in np.linspace(0, 1, 6):
        x = left + t * (w-left-right); xv = x_min + t*(x_max-x_min)
        y = h-bottom - t * (h-top-bottom); yv = y_min + t*(y_max-y_min)
        lines.append(f'<text x="{x}" y="{h-bottom+22}" text-anchor="middle" font-family="Arial" font-size="12">{xv:.3g}</text>')
        lines.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" font-family="Arial" font-size="12">{yv:.3g}</text>')
        lines.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{h-bottom}" stroke="#eee"/>')
        lines.append(f'<line x1="{left}" y1="{y}" x2="{w-right}" y2="{y}" stroke="#eee"/>')
    for label, xarr, yarr, color in series:
        pts = ' '.join(f'{px(float(x)):.2f},{py(float(y)):.2f}' for x, y in zip(xarr, yarr))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{pts}"/>')
    lx = left + 20
    for i, (label, _, _, color) in enumerate(series):
        ly = top + 25 + i*24
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+30}" y2="{ly}" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="{lx+40}" y="{ly+5}" font-family="Arial" font-size="14">{label}</text>')
    lines.append(f'<text x="{w/2}" y="{h-25}" text-anchor="middle" font-family="Arial" font-size="15">{x_label}</text>')
    lines.append(f'<text x="20" y="{h/2}" text-anchor="middle" transform="rotate(-90 20 {h/2})" font-family="Arial" font-size="15">{y_label}</text>')
    lines.append('</svg>')
    path.write_text('\n'.join(lines))


def svg_hist(path: Path, title: str, genuine: np.ndarray, impostor: np.ndarray, x_label: str) -> None:
    bins = np.linspace(min(genuine.min(), impostor.min()), max(genuine.max(), impostor.max()), 70)
    g_counts, edges = np.histogram(genuine, bins=bins, density=True)
    i_counts, _ = np.histogram(impostor, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    svg_line_plot(path, title, [("Genuine", centers, g_counts, "#2563eb"), ("Impostor", centers, i_counts, "#dc2626")], x_label, "Density")


def write_outputs(evaluations: list[Evaluation]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in evaluations:
        svg_hist(PLOTS_DIR / f"{e.name}_distributions.svg", f"{e.name.title()} Genuine vs Impostor Distribution", e.genuine, e.impostor, "Score")
        svg_line_plot(PLOTS_DIR / f"{e.name}_far_frr.svg", f"{e.name.title()} FAR and FRR", [("FAR", e.thresholds, e.far, "#dc2626"), ("FRR", e.thresholds, e.frr, "#2563eb")], "Threshold", "Error Rate", force_unit_y=True)
        svg_line_plot(PLOTS_DIR / f"{e.name}_roc.svg", f"{e.name.title()} ROC Curve", [("ROC", e.fpr, e.tpr, "#16a34a")], "False Accept Rate", "True Accept Rate", force_unit_y=True)
        rows.append({
            "metric": e.name,
            "genuine_count": int(e.genuine.size),
            "impostor_count": int(e.impostor.size),
            "genuine_mean": float(e.genuine.mean()),
            "genuine_std": float(e.genuine.std(ddof=1)),
            "impostor_mean": float(e.impostor.mean()),
            "impostor_std": float(e.impostor.std(ddof=1)),
            "eer": e.eer,
            "eer_percent": e.eer * 100,
            "eer_threshold": e.eer_threshold,
            "decidability_index": e.decidability_index,
        })
    columns = list(rows[0])
    csv_lines = [",".join(columns)]
    for row in rows:
        csv_lines.append(",".join(str(row[column]) for column in columns))
    (RESULTS_DIR / "metrics_summary.csv").write_text("\n".join(csv_lines) + "\n")
    (RESULTS_DIR / "metrics_summary.json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    features = load_features(DATA_PATH)
    evaluations = []
    for metric in ("euclidean", "cosine"):
        genuine, impostor, lower_is_match = score_samples(features, metric)
        evaluations.append(compute_curves(metric, genuine, impostor, lower_is_match))
    write_outputs(evaluations)
    for e in evaluations:
        print(f"{e.name}: EER={e.eer*100:.2f}% threshold={e.eer_threshold:.6f} d'={e.decidability_index:.4f}")


if __name__ == "__main__":
    main()
