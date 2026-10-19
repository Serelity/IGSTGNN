"""Read-only GW9 SUM and completed, failed GW10 lookup provenance for GW11."""
import gc
import json
from pathlib import Path

import numpy as np

from experiments.branch_readout_gw9 import readout_auth as auth
from experiments.conditional_residual_gw10 import run_probe as gw10
from experiments.conditional_residual_gw10 import median_probe
from experiments.conditional_residual_gw10 import screen as screening


SCHEMA = 'gw11_authenticated_gw10_failed_probe_v1'
PINNED_VALIDATION_MAE = {'ORIGIN':13.288735464, 'OWN4':13.246264357, 'JOINT':13.261324009}
PIN_ATOL = 5e-9


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _capture(path, digests, expected=None):
    path = Path(path).resolve()
    actual = auth.sha256(path)
    auth.require(expected is None or actual == expected, 'GW10 source file digest differs: '+str(path))
    auth.require(digests.setdefault(str(path),actual) == actual, 'GW10 source changed during certification')
    return path


def _records_equal(actual, expected):
    return set(actual) == set(expected) and all(set(actual[arm]) == set(expected[arm])
        and all((value is None and expected[arm][key] is None)
            or (value is not None and expected[arm][key] is not None
                and np.isclose(value,expected[arm][key],atol=1e-9,rtol=1e-10))
            for key,value in actual[arm].items()) for arm in actual)


def verify_cache_bindings(recorded,current,digests):
    """Allow identical exporter code in another checkout, preserving cache paths."""
    auth.require(isinstance(recorded,dict) and len(recorded)==len(current), 'GW10 protocol cache inputs differ')
    unmatched=dict(recorded)
    for filename,expected in current.items():
        if filename in unmatched:
            auth.require(unmatched.pop(filename)==expected,'GW10 cache input digest differs')
            continue
        relative=next((name for name in gw10.export_cache.SOURCE_FILES
            if Path(filename).as_posix().endswith('/'+name)),None)
        auth.require(relative is not None,'GW10 immutable cache artifact path differs')
        matches=[path for path,value in unmatched.items() if value==expected
            and Path(path).as_posix().endswith('/'+relative)]
        auth.require(len(matches)==1,'GW10 exporter source path/hash binding differs: '+relative)
        _capture(matches[0],digests,expected);unmatched.pop(matches[0])
    auth.require(not unmatched,'Unmatched GW10 protocol cache source files')


def certify_gw10(directory, archive, certificate, cache, cache_metadata, cache_digests):
    """Recompute fixed GW10 validation records without fitting or promoting a lookup."""
    directory = Path(directory).resolve()
    digests = dict(cache_digests)
    analysis_path = _capture(directory/'analysis.json',digests)
    protocol_path = _capture(directory/'protocol.json',digests)
    screen_path = _capture(directory/'screening.json',digests)
    analysis,protocol,decision = map(_json,(analysis_path,protocol_path,screen_path))
    auth.require(analysis.get('schema') == gw10.SCHEMA and analysis.get('protocol') == gw10.PROTOCOL
        and analysis.get('source_certificate') == certificate
        and analysis.get('cache_metadata_sha256') == cache_digests[str(Path(cache_metadata['_directory'])/'cache.json')]
        and analysis.get('cache_runtime') == cache_metadata['runtime']
        and analysis.get('model_updates') == analysis.get('SUM_head_updates') == analysis.get('validation_fits') == 0
        and analysis.get('test_evaluated') is False, 'Expected the completed GW10 train-only residual probe')
    auth.require(protocol.get('schema')==gw10.SCHEMA and protocol.get('protocol')==gw10.PROTOCOL
        and protocol.get('source_identity_sha256')==certificate['source_identity_sha256'],
        'GW10 fixed protocol or source identity differs')
    verify_cache_bindings(protocol.get('input_digests'),cache_digests,digests)
    auth.require(analysis.get('screening') == decision
        and decision.get('status') == 'GW10_RESIDUAL_PROBE_SCREEN_COMPLETE'
        and decision.get('candidate') == 'JOINT' and decision.get('eligible_for_new_model_design') is False
        and decision.get('GW9_joint_training_authorized') is False
        and decision.get('candidate_for_confirmation') is False
        and decision.get('test_evaluated') is False, 'GW10 must retain its failed fixed JOINT screen')
    auth.require(all(isinstance(analysis.get(key),str) and len(analysis[key])==64
        for key in ('model_sha256','audit_model_sha256')), 'GW10 fitted model hashes missing')
    model_path = _capture(directory/'model.json',digests,analysis['model_sha256'])
    _capture(directory/'audit_model.json',digests,analysis.get('audit_model_sha256'))
    fitted = _json(_capture(directory/'fitted_before_validation.json',digests))
    auth.require(fitted == dict(model_sha256=digests[str(model_path)],
        protocol_sha256=digests[str(protocol_path)],validation_labels_used_for_fit=False),
        'GW10 mapping was not bound before validation scoring')
    model = median_probe.load_model(model_path)
    validation = cache['val']
    corrections = median_probe.predict_corrections(model,validation['features'],horizon=12)
    predictions = {'SUM':validation['pred'].astype(np.float64)}
    predictions.update({arm:predictions['SUM']+correction for arm,correction in corrections.items()})
    del corrections; gc.collect()
    records = gw10.records(predictions,validation)
    auth.require(_records_equal(records,analysis.get('validation',{})), 'GW10 validation records do not replay')
    for arm,expected in PINNED_VALIDATION_MAE.items():
        auth.require(abs(records[arm]['mae']-expected) <= PIN_ATOL,
            'GW11 requires the returned GW10 '+arm+' validation MAE')
    masks = dict(high_value_ge314=validation['y']>=314,
        long_horizon_7_12=np.arange(12)[None,:,None]>=6,
        flow_50_200=(validation['y']>=50)&(validation['y']<200))
    groups = {name:gw10.records(predictions,validation,mask) for name,mask in masks.items()}
    auth.require(set(analysis.get('groups',{})) == set(groups)
        and all(_records_equal(records,analysis['groups'][name]) for name,records in groups.items()),
        'GW10 protected group records do not replay')
    recomputed = screening.screen(records,analysis['training_order_audit'],groups)
    auth.require(recomputed == decision, 'GW10 locked screen does not match its bound records')
    del predictions; gc.collect()
    result = dict(schema=SCHEMA,directory=str(directory),source_identity_sha256=certificate['source_identity_sha256'],
        source_base='GW9_SUM_CONTROL_BEST',historical_controls_promoted_to_base=False,
        file_digests=digests,validation_records=records,validation_groups=groups,
        pinned_validation_mae=PINNED_VALIDATION_MAE,pin_absolute_tolerance=PIN_ATOL,
        model_sha256=analysis['model_sha256'],screening=decision,
        eligible_for_new_model_design=False,new_lookup_fits=0,new_neural_inference_calls=0,
        lookup_validation_recomputed=True,training_order_audit_metrics_recomputed=False,
        source_artifacts_written=False,test_evaluated=False)
    assert_unchanged(result)
    return result


def _export_repo_from_cache(cache_dir):
    """Locate the recorded execution checkout; the hard-pinned reader verifies it."""
    metadata=_json(Path(cache_dir).resolve()/'cache.json')
    recorded=metadata.get('source_certificate',{}).get('file_digests',{})
    suffix='/experiments/branch_readout_gw9/probe.py'
    matches=[Path(filename).resolve() for filename in recorded if Path(filename).as_posix().endswith(suffix)]
    auth.require(len(matches)==1,'Cache must preserve one recorded GW9 execution checkout')
    return matches[0].parents[2]


def load_inputs(probe_dir, cache_dir, gw10_dir):
    # Certificates contain absolute execution paths. Recreate the original
    # byte-checked certificate instead of rewriting it for the GW11 checkout.
    export_repo=_export_repo_from_cache(cache_dir)
    archive,last,certificate = auth.load_authenticated_run(probe_dir,repo=export_repo)
    del last
    cache,metadata,digests = gw10.load_cache(cache_dir,archive,certificate)
    # Supply location internally without changing the immutable cache metadata
    # returned to callers or embedded in earlier source certificates.
    located = dict(metadata,_directory=str(Path(cache_dir).resolve()))
    reference = certify_gw10(gw10_dir,archive,certificate,cache,located,digests)
    auth.assert_unchanged(certificate)
    return archive,certificate,cache,metadata,digests,reference


def assert_unchanged(reference):
    for filename,expected in reference['file_digests'].items():
        auth.require(auth.sha256(filename) == expected, 'Certified GW10/cache source changed: '+filename)


def require_output_outside_sources(output,probe_dir,certificate,cache_dir,gw10_dir,data_dir=None):
    path = auth.require_output_outside_sources(output,probe_dir,certificate)
    protected = [Path(cache_dir).resolve(),Path(gw10_dir).resolve()]
    if data_dir is not None: protected.append(Path(data_dir).resolve())
    for source in protected:
        auth.require(path != source and source not in path.parents and path not in source.parents,
            'GW11 output must be outside and must not contain immutable cache/GW10/data sources')
    return path
