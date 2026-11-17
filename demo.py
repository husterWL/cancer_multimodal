"""Synthetic interface demonstration; outputs are not manuscript results."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'models' / 'shared'))
from model import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'sample_data')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'demo')
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(20261104)
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.data / 'synthetic_features.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    clinical = torch.from_numpy(arrays['clinical'])
    model = Model(clinical.shape[1], 'both', 'concat')
    features = [torch.from_numpy(arrays[name]) for name in
                ('patches', 'patch_mask', 'clinical', 'evidence', 'gates')]
    labels = torch.from_numpy(arrays['labels']).float()
    train_mask = arrays['folds'] != 1
    test_mask = ~train_mask

    def inputs(selected):
        mask = torch.from_numpy(selected)
        patches, patch_mask, clinical_values, evidence, gates = features
        return [patches[mask][patch_mask[mask]], patch_mask[mask],
                clinical_values[mask], evidence[mask], gates[mask]]

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    losses = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(*inputs(train_mask)), labels[torch.from_numpy(train_mask)])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():
        details = model(*inputs(test_mask), detail=True)
        probabilities = details['logit'].sigmoid()
        zero_gate_inputs = inputs(test_mask)
        zero_gate_inputs[-1] = torch.zeros_like(zero_gate_inputs[-1])
        zero_details = model(*zero_gate_inputs, detail=True)
        assert torch.equal(zero_details['concept'], torch.zeros_like(zero_details['concept']))
    checkpoint = args.output / 'synthetic_model.pt'
    torch.save({'state_dict': model.state_dict(), 'clinical_dim': clinical.shape[1],
                'clinical': 'both', 'image': 'concat', 'synthetic': True}, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    restored = Model(saved['clinical_dim'], saved['clinical'], saved['image']).eval()
    restored.load_state_dict(saved['state_dict'])
    with torch.inference_mode():
        assert torch.equal(restored(*inputs(test_mask)).sigmoid(), probabilities)
    with (args.output / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['patient_id', 'fold', 'label', 'probability'])
        for identifier, label, probability in zip(arrays['patient_ids'][test_mask],
                                                  arrays['labels'][test_mask], probabilities.tolist()):
            writer.writerow([identifier, 1, int(label), probability])
    np.savez_compressed(args.output / 'attention.npz',
                        **{name: details[name].numpy() for name in
                           ('patch_attention', 'image_attention', 'concept_attention')})
    report = {'synthetic': True, 'train_patients': int(train_mask.sum()),
              'test_patients': int(test_mask.sum()), 'epochs': args.epochs,
              'training_loss': losses, 'checkpoint_roundtrip': True,
              'zero_gate_check': True, 'scope': 'Interface check, not clinical validation'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()