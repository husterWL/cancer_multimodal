"""Evaluate fixed patient-level OOF predictions; no model retraining."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def read_predictions(path):
    frame = pd.read_csv(path, dtype={'patient_id': str})
    if 'split' in frame:
        frame = frame.loc[frame['split'] == 'test'].copy()
    required = {'patient_id', 'fold', 'label', 'probability'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('Expected nonempty held-out patient_id,fold,label,probability rows')
    if frame[list(required)].isna().any().any() or frame.patient_id.duplicated().any():
        raise ValueError('Missing values or duplicate patient IDs')
    if set(frame.label) != {0, 1} or set(frame.fold) != {1, 2, 3, 4, 5}:
        raise ValueError('Expected both classes and all five held-out folds')
    if not np.isfinite(frame.probability).all() or not frame.probability.between(0, 1).all():
        raise ValueError('Probabilities must be finite and in [0, 1]')
    return frame.sort_values('patient_id').reset_index(drop=True)


def metrics(labels, probabilities):
    predicted = probabilities >= 0.5
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predicted, labels=[0, 1]).ravel()

    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else float('nan')

    return dict(AUC=float(roc_auc_score(labels, probabilities)),
                ACC=float(np.mean(predicted == labels)),
                Precision=divide(true_positive, true_positive + false_positive),
                Recall=divide(true_positive, true_positive + false_negative),
                Specificity=divide(true_negative, true_negative + false_positive),
                F1=float(f1_score(labels, predicted, zero_division=0)),
                NPV=divide(true_negative, true_negative + false_negative))


def paired_tests(labels, first, second):
    positive = labels == 1
    negative = ~positive
    if min(positive.sum(), negative.sum()) < 2:
        raise ValueError('DeLong covariance needs at least two patients in each class')
    positive_placements, negative_placements = [], []
    for probability in (first, second):
        difference = probability[positive, None] - probability[None, negative]
        comparison = (difference > 0).astype(float) + 0.5 * (difference == 0)
        positive_placements.append(comparison.mean(axis=1))
        negative_placements.append(comparison.mean(axis=0))
    covariance = (np.cov(positive_placements, ddof=1) / positive.sum()
                  + np.cov(negative_placements, ddof=1) / negative.sum())
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    delta = float(roc_auc_score(labels, first) - roc_auc_score(labels, second))
    delong = float(2 * norm.sf(abs(delta) / np.sqrt(variance))) if variance > 0 else None
    first_correct = (first >= 0.5) == labels
    second_correct = (second >= 0.5) == labels
    only_first = int((first_correct & ~second_correct).sum())
    only_second = int((~first_correct & second_correct).sum())
    mcnemar = float(binomtest(only_first, only_first + only_second, 0.5).pvalue) if only_first + only_second else 1.0
    return dict(auc_difference=delta, delong_p_two_sided=delong,
                delong_status='ok' if variance > 0 else 'undefined: zero difference variance',
                mcnemar_exact_p_two_sided=mcnemar, only_first_correct=only_first,
                only_second_correct=only_second)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error('--bootstrap must be at least 2')
    frame = read_predictions(args.predictions)
    labels = frame.label.to_numpy(dtype=int)
    probabilities = frame.probability.to_numpy(dtype=float)
    point = metrics(labels, probabilities)
    generator = np.random.default_rng(args.seed)
    sampled = {name: [] for name in point}
    accepted = 0
    for iteration in range(args.bootstrap):
        indices = generator.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        for name, value in metrics(labels[indices], probabilities[indices]).items():
            if np.isfinite(value):
                sampled[name].append(value)
        accepted += 1
    intervals = {name: dict(estimate=point[name] if np.isfinite(point[name]) else None, ci95=np.quantile(values, [0.025, 0.975]).tolist()
                            if values else None, valid_resamples=len(values))
                 for name, values in sampled.items()}
    report = dict(patients=len(labels), threshold=0.5, bootstrap_requested=args.bootstrap,
                  bootstrap_accepted=accepted, bootstrap_seed=args.seed, metrics=intervals,
                  interval_scope='Non-stratified patient percentile bootstrap conditional on fixed OOF predictions; excludes retraining uncertainty',
                  provenance='Review utility added during packaging; not asserted to be the historical paper statistics program')
    if args.compare:
        other = read_predictions(args.compare)
        if not frame[['patient_id', 'label', 'fold']].equals(other[['patient_id', 'label', 'fold']]):
            raise ValueError('Paired tests require identical patient IDs, labels and held-out folds')
        report['paired_tests'] = paired_tests(labels, probabilities, other.probability.to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()