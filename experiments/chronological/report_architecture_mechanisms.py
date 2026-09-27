"""Print the small, shareable v12a report without importing PyTorch."""

import argparse
import json
from pathlib import Path


def report(path):
    summary = json.loads(Path(path).read_text(encoding='utf-8'))
    if summary.get('protocol_id') != 'contra_v8_architecture_mechanism_audit_v12a':
        raise ValueError('Expected a v12a architecture audit summary')
    print('status:', summary['status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    for key in ('engineering_check', 'full_cohort_evaluated', 'checkpoint_state_unchanged',
                'model_training_performed', 'real_data_gradient_computation_performed',
                'validation_arrays_read', 'test_split_read', 'recommendation'):
        print(f'{key}:', summary[key])
    print('\nSYNTHETIC ICSF GRADIENTS (module copy, not full-model training)')
    for name, value in summary['synthetic']['gradient_abs_max'].items():
        if name.startswith(('q_proj.', 'k_proj.', 'v_proj.', 'icsf_fusion_mlp.')):
            print(name, value)
    for name in ('semantic_attention_vs_mask', 'pre_norm_injection_vs_mask_times_V',
                 'zero_support_latest_state_change', 'zero_support_vs_normalization_only'):
        print(name, summary['synthetic'][name])
    for cohort, result in summary['results'].items():
        print(f'\n[{cohort}] samples={result["samples"]}')
        print('descriptive_all_MAE:', json.dumps(result['regions']['all']['descriptive_mae']))
        mechanisms = result['mechanisms_standardized']
        for key in ('native_full_replay', 'native_off_replay', 'normalization_non_candidate',
                    'icsf_vs_norm_non_candidate', 'dynamic_graph_icsf_vs_norm'):
            print(key, mechanisms[key])
        for region in ('candidate_h1_h3', 'candidate_h4_h6', 'noncandidate_h1_h6',
                       'candidate_h7_h12'):
            metrics = result['regions'][region]['prediction_sensitivity']
            print(region, 'raw_delta_abs_mean:', json.dumps({
                name: metrics[name]['abs_mean'] for name in (
                    'normalization', 'icsf_given_normalization', 'tiid_given_icsf',
                    'icsf_tiid_interaction', 'dynamic_graph_replay_sensitivity')}))
    print('\nInterpretation:', summary['interpretation'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    report(parser.parse_args().summary)
