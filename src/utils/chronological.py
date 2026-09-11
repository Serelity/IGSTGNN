"""Explicit masks and report-only inputs for the conditional chronology experiment."""

import csv
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def valid_flow(values):
    return np.isfinite(values) & (values >= 0)


def encode_window(raw, x_start, scaler):
    """The 26 raw slots contain X[0:12], a two-slot gap, and Y[14:26]."""
    if raw.ndim != 2 or raw.shape[0] != 26:
        raise ValueError('Expected raw flow [26, nodes]')
    history, target = raw[:12], raw[14:26]
    x_mask, y_mask = valid_flow(history), valid_flow(target)
    fill = np.asarray(scaler['node_fill_mean'], dtype=np.float64)
    mean, std = float(scaler['mean']), float(scaler['std'])
    if fill.shape != (raw.shape[1],) or not np.isfinite(fill).all():
        raise ValueError('Training node-fill statistics do not match the node axis')
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid training scaler')
    x = np.empty((12, raw.shape[1], 3), dtype=np.float32)
    x[..., 0] = (np.where(x_mask, history, fill) - mean) / std
    for step in range(12):
        time = x_start + timedelta(minutes=5 * step)
        x[step, :, 1] = (time.hour * 12 + time.minute // 5) / 288
        x[step, :, 2] = ((time.weekday() + 1) % 7) / 7
    y = np.where(y_mask, target, 0).astype(np.float32)[..., None]
    return x, y, x_mask[..., None], y_mask[..., None]


class ChronologicalDataset(Dataset):
    def __init__(self, data_dir, split):
        if split not in ('train', 'val'):
            raise ValueError('Only the development train/val package is supported')
        self.directory = Path(data_dir)
        self.rows = read_rows(self.directory / f'{split}_manifest.csv')
        self.flow = np.load(self.directory / f'{split}_flow.npy', mmap_mode='r', allow_pickle=False)
        self.station_ids = np.load(self.directory / 'station_ids.npy', allow_pickle=False)
        self.scaler = json.loads((self.directory / 'scaler.json').read_text())
        with np.load(self.directory / f'{split}_context.npz', allow_pickle=False) as stored:
            self.context = {k: stored[k].copy() for k in stored.files}
        indices = [int(r['sample_index']) for r in self.rows]
        if self.flow.shape != (len(self.rows), 26, len(self.station_ids)):
            raise ValueError('Flow shape and manifest disagree')
        if not np.array_equal(self.context['sample_indices'], indices):
            raise ValueError('Event context and traffic sample order disagree')
        if not np.array_equal(self.context['station_ids'], self.station_ids):
            raise ValueError('Event context and traffic station order disagree')
        if self.context['distances'].shape != (len(self.rows), len(self.station_ids), 3):
            raise ValueError('Event spatial features have an unexpected shape')
        if not np.isfinite(self.context['distances']).all():
            raise ValueError('Non-finite event spatial features')
        if any(row['split'] != split or int(row['source_version']) != 8 for row in self.rows):
            raise ValueError('Manifest has mixed splits or traffic versions')
        if (self.scaler.get('source_version') != 8 or self.scaler.get('fit_scope') !=
                'train_X_0:12_unique_station_nominal_slot_finite_nonnegative'):
            raise ValueError('Expected full-package statistics fitted only on unique training X')
        if not np.array_equal(self.scaler['station_ids'], self.station_ids):
            raise ValueError('Scaler and raw flow station order disagree')
        train_rows = read_rows(self.directory / 'train_manifest.csv')
        if self.scaler['fitted_sample_indices'] != [int(r['sample_index']) for r in train_rows]:
            raise ValueError('Scaler fitting rows differ from the training manifest')
        for field in ('report_age_minutes', 'forecast_tod', 'forecast_dow'):
            if self.context[field].shape != (len(self.rows),) or not np.isfinite(self.context[field]).all():
                raise ValueError(f'Invalid context vector: {field}')
        for i, row in enumerate(self.rows):
            report = datetime.fromisoformat(row['report_time'])
            t0 = datetime.fromisoformat(row['t0'])
            elapsed = (t0 - report).total_seconds() / 60
            if not 0 < elapsed <= 5 or abs(float(self.context['report_age_minutes'][i]) - elapsed) > 1e-5:
                raise ValueError('Context report age disagrees with manifest')
            if (self.context['forecast_tod'][i] != t0.hour * 12 + t0.minute // 5 or
                    self.context['forecast_dow'][i] != (t0.weekday() + 1) % 7):
                raise ValueError('Context forecast clock disagrees with manifest')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        x, y, x_mask, y_mask = encode_window(
            self.flow[index], datetime.fromisoformat(self.rows[index]['x_start']), self.scaler)
        return {'x': x, 'y_flow': y, 'x_valid': x_mask, 'y_valid': y_mask,
                'incident': {k: self.context[k][index] for k in
                             ('report_age_minutes', 'forecast_tod', 'forecast_dow', 'distances')}}


def masked_flow_mae(prediction, target, valid):
    """Pooled training MAE; invalid targets never become valid zero-flow labels."""
    if prediction.shape != target.shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('Prediction, target and explicit bool mask must have equal shapes')
    if not valid.any():
        raise ValueError('Batch contains no valid forecast targets')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Non-finite prediction or valid target')
    return (prediction[valid] - target[valid]).abs().mean()


def flow_metrics(prediction, target, valid):
    """Separate horizon macro averages from pooled metrics and positive-flow MAPE."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid)
    if prediction.shape != target.shape or valid.shape != target.shape or valid.dtype != np.bool_:
        raise ValueError('Metrics require equal shapes and an explicit boolean mask')
    if prediction.ndim != 4 or not np.isfinite(prediction).all() or not np.isfinite(target[valid]).all():
        raise ValueError('Invalid metric input')
    absolute, squared, counts, percent, positive_counts = [], [], [], [], []
    for h in range(target.shape[1]):
        selected = valid[:, h]
        diff = prediction[:, h][selected] - target[:, h][selected]
        positive = selected & (target[:, h] > 0)
        absolute.append(float(np.abs(diff).sum()))
        squared.append(float(np.square(diff).sum()))
        counts.append(int(selected.sum()))
        percent.append(float(np.sum(np.abs(prediction[:, h][positive] - target[:, h][positive])
                                    / target[:, h][positive])))
        positive_counts.append(int(positive.sum()))
    if any(n == 0 for n in counts):
        raise ValueError('At least one horizon has no valid targets; macro metric is undefined')
    mae = np.asarray(absolute) / counts
    rmse = np.sqrt(np.asarray(squared) / counts)
    mape = [a / n if n else None for a, n in zip(percent, positive_counts)]
    return {'mae_macro': float(mae.mean()), 'rmse_macro': float(rmse.mean()),
            'mape_macro': float(np.mean(mape)) if all(n > 0 for n in positive_counts) else None,
            'mae_pooled': sum(absolute) / sum(counts),
            'rmse_pooled': float(np.sqrt(sum(squared) / sum(counts))),
            'mape_pooled': sum(percent) / sum(positive_counts) if sum(positive_counts) else None,
            'per_horizon_mae': mae.tolist(), 'per_horizon_rmse': rmse.tolist(),
            'per_horizon_mape': mape, 'valid_count_per_horizon': counts,
            'positive_target_count_per_horizon': positive_counts}
