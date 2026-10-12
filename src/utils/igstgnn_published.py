"""Read the author model-ready release without rebuilding or normalizing it.

The all-sample object array necessarily deserializes test payloads. Only the
requested split is batched; training never evaluates test or fits statistics.
"""
import csv
from pathlib import Path

import numpy as np

from data.xtraffic.prepare_splits import load_samples
from src.utils.dataloader import IncidentDataLoader, StandardScaler


def require(condition, message):
    if not condition:
        raise ValueError(message)


class PublishedLoader(IncidentDataLoader):
    """Original batch construction, with a resumable sample permutation."""
    def __init__(self, samples, batch_size, logger):
        self.original_samples = tuple(samples)
        self.order = np.arange(len(samples), dtype=np.int64)
        super().__init__(samples, False, batch_size, logger, input_dim=3)

    def set_order(self, order):
        order = np.asarray(order, dtype=np.int64)
        require(order.shape == (self.size,) and np.array_equal(np.sort(order), np.arange(self.size)),
                'Invalid published training permutation')
        self.order = order.copy()
        self.samples = [self.original_samples[i] for i in order]

    def shuffle(self):
        self.set_order(self.order[np.random.permutation(self.size)])


class PublishedData:
    def __init__(self, directory, protocol, logger, *, check=False):
        directory = Path(directory)
        # Keep exact author order and prepare_splits.py rounding, including the
        # remainder in test. No new date assignment or v8 replacement occurs.
        samples = load_samples(directory/'incident_all.npy')
        total = len(samples)
        train_end = int(total*.70)
        val_end = train_end+int(total*.15)
        counts = dict(total_samples=total, train_samples=train_end,
                      val_samples=val_end-train_end, test_samples=total-val_end)
        require(all(protocol[key] == value for key, value in counts.items()),
                'Published sample counts differ from frozen release')
        with (directory/'sensors.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.station_ids = [int(row['station_id']) for row in rows]
        require(len(self.station_ids) == protocol['stations'] and len(set(self.station_ids)) == len(rows),
                'Published station axis mismatch')
        # Validate every train/val sample; test targets are not inspected here.
        for i, sample in enumerate(samples[:val_end]):
            for key in ('x_data', 'y_data'):
                value = np.asarray(sample[key])
                expected = (12, protocol['stations'], 3)
                require(value.shape == expected and value.dtype == np.float32 and np.isfinite(value).all(),
                        f'Invalid published {key} at sample {i}')
            clocks = sample['x_data'][..., 1:]
            require(np.array_equal(clocks, np.broadcast_to(clocks[:, :1], clocks.shape)),
                    f'Published history clocks disagree across stations: {i}')
            nominal = clocks[:, 0]*np.array([288, 7], dtype=np.float32)
            require(np.all(np.abs(nominal-np.round(nominal)) <= 3e-5)
                    and np.all((nominal >= 0) & (nominal < [288, 7])),
                    f'Published history clocks are invalid: {i}')
        with np.load(directory/'incident_stats.npz', allow_pickle=False) as stats:
            require(bool(stats['normalized']), 'Expected already-normalized author release')
            mean, std = float(stats['mean']), float(stats['std'])
        require(np.isfinite(mean) and np.isfinite(std) and std > 0, 'Invalid published scaler')
        self.scaler = StandardScaler(mean, std)
        batch_size = 2 if check else protocol['batch_size']
        self.loaders = {
            'train_loader': PublishedLoader(samples[:min(train_end, 4) if check else train_end], batch_size, logger),
            'val_loader': PublishedLoader(samples[train_end:min(val_end, train_end+4) if check else val_end], batch_size, logger),
        }
        self._test_samples = samples[val_end:]
        self.audit = dict(data_source='author_model_ready_release', split_policy='author_order_70_15_15',
                          counts=counts, used_counts={key: loader.size for key, loader in self.loaders.items()},
                          normalization='published_values_and_incident_stats_unchanged', mean=mean, std=std,
                          x_slice='published_x_data_all_12_steps_all_3_channels',
                          y_slice='published_y_data_all_12_steps_first_flow_channel',
                          clock_offset_slots=0, traffic_rebuilt=False, scaler_refitted=False,
                          incident_inputs_used=False, test_payload_deserialized=True, test_evaluated=False,
                          absolute_calendar_order_independently_verified=False)

    def test_loader(self, batch_size, logger):
        return PublishedLoader(self._test_samples, batch_size, logger)
