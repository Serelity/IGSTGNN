"""Strict, resumable, bounded byte-range reads for the fixed v8 source."""
import ast
import calendar
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import requests
from urllib3.exceptions import HTTPError


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


class RangeProtocolError(ValueError):
    pass


def validate_range_headers(status, headers, start, end, total=None):
    if status != 206:
        raise RangeProtocolError(f'HTTP {status}; required 206')
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', headers.get('Content-Range', ''))
    if not match or tuple(map(int, match.groups()[:2])) != (start, end):
        raise RangeProtocolError('Content-Range endpoints differ')
    reported_total = int(match.group(3))
    if reported_total <= end or (total is not None and reported_total != total):
        raise RangeProtocolError('Content-Range total differs')
    length = headers.get('Content-Length')
    if length is not None and int(length) != end - start + 1:
        raise RangeProtocolError('Content-Length differs')


def parse_header(payload, month):
    if payload[:6] != b'\x93NUMPY' or tuple(payload[6:8]) not in {(1, 0), (2, 0), (3, 0)}:
        raise ValueError('Invalid NPY header/version')
    prefix = 10 if payload[6] == 1 else 12
    offset = prefix + int.from_bytes(payload[8:prefix], 'little')
    if offset > len(payload):
        raise ValueError('Incomplete NPY header')
    header = ast.literal_eval(payload[prefix:offset].decode('latin1'))
    expected_shape = (16972, calendar.monthrange(2023, month)[1] * 288, 3)
    if header['shape'] != expected_shape or header['descr'] != '<f4' or header['fortran_order']:
        raise ValueError('Expected v8 node-major float32 monthly source')
    return {'shape': list(expected_shape), 'dtype': '<f4', 'data_offset': offset,
            'total_bytes': offset + expected_shape[0] * expected_shape[1] * 12}


class RowCache:
    """Request identities point to SHA-addressed blobs; both are checked on reuse."""
    def __init__(self, directory):
        self.directory = Path(directory)

    def read(self, identity, fetch):
        encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
        key = hashlib.sha256(encoded).hexdigest()
        request_dir = self.directory / 'requests'
        blob_dir = self.directory / 'blobs'
        request_dir.mkdir(parents=True, exist_ok=True)
        blob_dir.mkdir(parents=True, exist_ok=True)
        record_path = request_dir / (key + '.json')
        expected_bytes = identity['end'] - identity['start'] + 1
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding='utf-8'))
            blob = blob_dir / (record['sha256'] + '.bin')
            if record['identity'] != identity or not blob.exists():
                raise ValueError('Row cache identity or payload missing')
            payload = blob.read_bytes()
            if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != record['sha256']:
                raise ValueError('Row cache checksum/length mismatch')
            return payload, {**record, 'cache_hit': True}
        payload = fetch()
        if len(payload) != expected_bytes:
            raise RangeProtocolError('Payload byte length differs; not cached')
        digest = hashlib.sha256(payload).hexdigest()
        blob = blob_dir / (digest + '.bin')
        # Different all-zero source rows may share one content hash.
        temporary = blob_dir / (digest + '.' + key + '.partial')
        temporary.write_bytes(payload)
        temporary.replace(blob)
        record = {'identity': identity, 'bytes': len(payload), 'sha256': digest}
        save_json(record_path, record)
        return payload, {**record, 'cache_hit': False}


class V8Reader:
    endpoint = 'https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset'

    def __init__(self):
        self._stats = Counter()
        self._lock = threading.Lock()

    def add(self, name, value=1):
        with self._lock:
            self._stats[name] += value

    def stats(self):
        with self._lock:
            return dict(self._stats)

    def signed_url(self, filename):
        for attempt in range(4):
            self.add('signed_url_attempts')
            try:
                with requests.post(self.endpoint, json={
                    'owner_slug': 'gpxlcj', 'dataset_slug': 'xtraffic',
                    'file_name': filename, 'dataset_version_number': 8,
                }, allow_redirects=False, timeout=(10, 30)) as response:
                    if response.status_code != 302 or not response.headers.get('Location'):
                        raise RangeProtocolError(f'Kaggle HTTP {response.status_code}; required redirect')
                    return response.headers['Location']
            except (requests.RequestException, RangeProtocolError):
                self.add('signed_url_failures')
                if attempt == 3:
                    raise RuntimeError('Kaggle signed URL acquisition failed; URL suppressed') from None
                self.add('retries')
                time.sleep(2 ** attempt)

    def read_range(self, url, start, end, total=None):
        if start < 0 or end < start:
            raise ValueError('Invalid byte range')
        for attempt in range(4):
            self.add('range_attempts')
            try:
                with requests.get(url, headers={'Range': f'bytes={start}-{end}',
                                                'Accept-Encoding': 'identity'},
                                  stream=True, timeout=(10, 60)) as response:
                    validate_range_headers(response.status_code, response.headers, start, end, total)
                    response.raw.decode_content = False
                    payload = response.raw.read(end - start + 2)
                self.add('range_bytes_received', len(payload))
                if len(payload) != end - start + 1:
                    raise RangeProtocolError('Range body length differs')
                self.add('range_successes')
                return payload
            except (requests.RequestException, HTTPError, OSError, ValueError) as error:
                self.add('range_failures')
                if attempt == 3:
                    raise RuntimeError(f'Range failed after four attempts ({type(error).__name__}); URL suppressed') from None
                self.add('retries')
                time.sleep(2 ** attempt)

    def open_month(self, month, cache):
        if month not in range(1, 11):
            raise ValueError('Only January through October authorized; no test months')
        filename = f'year_2023/year_2023/2023_p{month:02d}.npy'
        url = self.signed_url(filename)
        identity = {'dataset': 'gpxlcj/xtraffic', 'version': 8, 'year': 2023,
                    'month': month, 'file_name': filename, 'start': 0, 'end': 511}
        payload, record = cache.read(identity, lambda: self.read_range(url, 0, 511))
        return {'month': month, 'filename': filename, 'url': url,
                'layout': parse_header(payload, month), 'header_record': record}

    def read_node(self, remote, raw_axis, cache):
        layout = remote['layout']
        if not 0 <= int(raw_axis) < layout['shape'][0]:
            raise ValueError('Raw node index outside source axis')
        size = layout['shape'][1] * 12
        start = layout['data_offset'] + int(raw_axis) * size
        identity = {'dataset': 'gpxlcj/xtraffic', 'version': 8, 'year': 2023,
                    'month': remote['month'], 'file_name': remote['filename'],
                    'raw_node_index': int(raw_axis), 'start': start, 'end': start + size - 1,
                    'header_sha256': remote['header_record']['sha256']}
        return cache.read(identity, lambda: self.read_range(remote['url'], start,
                                                           start + size - 1, layout['total_bytes']))
