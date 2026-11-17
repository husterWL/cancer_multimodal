"""Generate artificial features, never read private study data."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'sample_data')
    args = parser.parse_args()
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261104)
    count = 20
    identifiers = np.array(['SYN%04d' % (index + 1) for index in range(count)])
    folds = np.repeat(np.arange(1, 6), 4)
    labels = np.tile([0, 1, 0, 1], 5)
    available = np.ones(count, dtype=np.uint8)
    available[[0, 5, 10, 15]] = 0
    categorical = rng.integers(0, 3, (count, 28))
    base = np.eye(3, dtype=np.float32)[categorical].reshape(count, -1)
    knowledge = rng.normal(0, 0.1, (count, 512)).astype('float32') * available[:, None]
    clinical = np.concatenate([base, knowledge, available[:, None]], axis=1).astype('float32')
    patches = rng.normal(0, 0.1, (count, 2, 4, 512)).astype('float32')
    patch_mask = np.ones((count, 2, 4), dtype=bool)
    patch_mask[::2, 1, -1] = False
    evidence = rng.normal(0, 0.1, (count, 22, 512)).astype('float32')
    gates = rng.uniform(0, 1, (count, 22)).astype('float32')
    np.savez_compressed(destination / 'synthetic_features.npz', patient_ids=identifiers,
                        folds=folds, labels=labels, clinical=clinical, patches=patches,
                        patch_mask=patch_mask, evidence=evidence, gates=gates)
    rows = [dict(patient_id=str(identifier), label=int(labels[index]), fold=int(folds[index]),
                 **{'feature_%02d' % (column + 1): str(value)
                    for column, value in enumerate(categorical[index])})
            for index, identifier in enumerate(identifiers)]
    write_csv(destination / 'clinical.csv', rows)
    external = destination / 'outputs_layout'
    cohort = external / 'coordinate_patch_183_20261105'
    manifest = [dict(patient_id=str(identifier), label=int(labels[index]), fold=int(folds[index]),
                     kg_available_mask=int(available[index]), kg_missing_mask=int(1 - available[index]))
                for index, identifier in enumerate(identifiers)]
    write_csv(cohort / 'audit' / 'fixed_five_fold_manifest.csv', manifest)
    (external / 'audit').mkdir(parents=True, exist_ok=True)
    (external / 'audit' / 'raw_clinical_inputs.json').write_text(json.dumps({
        'synthetic': True, 'patients': {str(identifier): categorical[index].astype(str).tolist()
                                     for index, identifier in enumerate(identifiers)}}, indent=2))
    (cohort / 'patients').mkdir(parents=True, exist_ok=True)
    for index, identifier in enumerate(identifiers):
        values = dict(visual=patches[index], patch_mask=patch_mask[index],
                      image_paths=np.array(['synthetic_image_1', 'synthetic_image_2']),
                      coordinate_keys=np.array(['synthetic_1', 'synthetic_2']))
        for kind in ('name', 'concat'):
            values[kind + '_evidence'] = evidence[index] if kind == 'concat' else evidence[index] * 0.9
            values[kind + '_scores'] = rng.uniform(0.1, 0.4, 22).astype('float32')
            values[kind + '_agreement'] = rng.uniform(0, 1, 22).astype('float32')
        np.savez_compressed(cohort / 'patients' / (str(identifier) + '.npz'), **values)
    for kind in ('knowledge', 'random_gat', 'no_neighbor'):
        vectors = knowledge if kind == 'knowledge' else rng.normal(
            0, 0.1, (count, 768 if kind == 'no_neighbor' else 512)).astype('float32') * available[:, None]
        for fold in range(1, 6):
            folder = cohort / 'clinical' / kind / ('fold_%d' % fold)
            folder.mkdir(parents=True, exist_ok=True)
            for split, selected in [('train', folds != fold), ('test', folds == fold)]:
                np.savez_compressed(folder / (split + '_patient_embeddings.npz'),
                                    patient_ids=identifiers[selected], labels=labels[selected],
                                    embeddings=vectors[selected])
    metadata = {'synthetic': True, 'seed': 20261104, 'patients': count,
                'clinical_columns': 28, 'folds': 5, 'image_feature_dimension': 512,
                'concepts': 22, 'source': 'Independent pseudorandom numbers; no patient inputs',
                'warning': 'Not clinical images, encoder outputs, study folds or clinical evidence'}
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()