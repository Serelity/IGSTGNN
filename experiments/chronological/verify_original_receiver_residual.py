"""Server-only GW4 engineering checks; never re-evaluate a completed baseline.

Synthetic checks use optimizer steps on a tiny fixture. Optional real-release
checks perform one new-candidate MAE backward per arm, with zero real-data
optimizer updates. Neither mode is predictive-performance evidence.
"""
import argparse
import copy
import gc
import json
import logging
from pathlib import Path
import sys
import tempfile
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch
from torch import nn

from experiments.chronological import train_original_receiver_residual as train
from experiments.chronological.original_graph_message import GraphMessageIGSTGNN
from experiments.chronological.original_receiver_residual import ARMS, ReceiverResidualIGSTGNN, split_state
from experiments.chronological.receiver_residual_reference import certify, recertify, require
from experiments.chronological.train_original_isp import restore_rng


class FixtureConv(nn.Module):
    """Small native repeated-adjacency contract, including five sources/two orders."""
    def __init__(self, graphs, channels=4, kernel=3):
        super().__init__()
        self.hidden_dim, self.k_t, self.k_s = channels, kernel, 2
        self.use_predefined_graph = self.use_dynamic_hidden_graph = self.use_static_hidden_graph = True
        self.pre_defined_graph = self.get_graph(graphs[:2])
        self.num_matric = 11
        self.fc_list_updt = nn.Linear(kernel*channels, kernel*channels)
        self.activation = nn.Tanh()
        self.gcn_updt = nn.Linear(self.num_matric*channels, channels)
        self.dropout = nn.Dropout(.3)

    def get_graph(self, graphs):
        result = []
        for graph in graphs:
            power = graph
            for _ in range(self.k_s):
                result.append(power.repeat(1, self.k_t))
                power = power@graph
        return result


class FixtureOriginal(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.node_num, self.input_dim, self.output_dim, self.seq_len, self.horizon = 5, 4, 1, 7, 3
        graphs = [torch.softmax(torch.randn(5, 5, device=device), -1) for _ in range(5)]
        native = FixtureConv(graphs).to(device)
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].dif_layer = nn.Module()
        self.layers[0].dif_layer.localized_st_conv = native
        self.register_buffer('graph3', graphs[2])
        self.register_buffer('graph4', graphs[3])
        self.register_buffer('graph5', graphs[4])
        self.readout = nn.Linear(4, 1).to(device)

    def forward(self, history_data, incident_data=None, sensor_data=None):
        conv = self.layers[0].dif_layer.localized_st_conv
        native = conv.native
        dynamic = native.get_graph([self.graph3, self.graph4])
        static = [self.graph5]
        state = conv(history_data, dynamic, static)
        values, current = [state[:, -1:]], history_data[:, -2:]
        for _ in range(2):
            current = torch.cat((current[:, -2:], values[-1]), 1)
            values.append(conv(current, dynamic, static))
        return self.readout(torch.cat(values, 1))


def fixture(device):
    torch.manual_seed(1847)
    anchor = GraphMessageIGSTGNN(FixtureOriginal(device), 'POST_MESSAGE').eval()
    # Exercise a nonuniform learned anchor, not only the uniform initialization.
    with torch.no_grad():
        anchor.selector.router.static_logits.uniform_(-.3, .3)
        anchor.selector.router.score_head.weight.normal_(0., .2)
    x = torch.randn(2, 7, 5, 4, device=device)
    target = torch.randn(2, 3, 5, 1, device=device)
    return anchor, x, target


def synthetic_checks(device):
    anchor, x, target = fixture(device)
    source_hash = train.base.tensor_tree_hash(anchor.state_dict())
    input_hash = train.base.tensor_tree_hash((x, target))
    with torch.no_grad():
        expected = anchor(x)
    result = dict(recovery={}, gradients={}, interventions={}, optimizer_resume={},
        scope='synthetic_repeated_adjacency_and_autoregressive_fixture_not_native_release_replay')
    models, initial = {}, {}
    for arm in ARMS:
        torch.manual_seed(93)
        before = train.base.tensor_tree_hash(train.base.rng_state())
        model = ReceiverResidualIGSTGNN(copy.deepcopy(anchor), arm)
        require(before == train.base.tensor_tree_hash(train.base.rng_state()), 'GW4 construction changed RNG')
        initial[arm] = train.initialization(model)
        checks = []
        for training in (False, True):
            model.train(training)
            before = train.base.tensor_tree_hash(train.base.rng_state())
            with torch.no_grad():
                prediction = model(x, label=target*100.)
            require(torch.allclose(prediction, expected, atol=2e-6, rtol=2e-6), 'GW4 zero residual did not recover its anchor')
            require(before == train.base.tensor_tree_hash(train.base.rng_state()), 'Frozen dropout consumed RNG')
            require(not any(m.training for m in model.selector.native.modules()), 'Frozen native module entered train mode')
            checks.append(dict(training_requested=training, abs_max=float((prediction-expected).abs().max())))
        require(train.base.tensor_tree_hash(split_state(model.state_dict())[0]) == source_hash, 'Construction altered frozen state')
        result['recovery'][arm] = checks
        models[arm] = model
    for key in ('frozen_tensor_sha256', 'added_tensor_sha256', 'post_construction_rng_sha256'):
        require(len({row[key] for row in initial.values()}) == 1, 'Synthetic paired initialization differs')
    require(initial['MESSAGE_RESIDUAL']['trainable_parameters'] == initial['RECEIVER_RESIDUAL']['trainable_parameters'],
            'Receiver/message active capacity differs')
    require(initial['STATIC_RESIDUAL']['trainable_parameters'] == 15, 'Static control trains more than five-by-three biases')
    result['initializations'] = initial
    synthetic_steps = 0
    for arm, model in models.items():
        model.train()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.002, weight_decay=1e-5)
        rows = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x)-target).square().mean()
            loss.backward()
            require(bool(torch.isfinite(loss)), 'Nonfinite synthetic loss')
            require(all(p.grad is None for name, p in model.named_parameters() if '.residual.' not in name),
                    'Frozen parameter received task gradient')
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                    'Nonfinite GW4 gradient')
            residual = model.selector.residual
            key_norm = float(residual.key.weight.grad.square().sum()) if residual.key.weight.grad is not None else 0.
            query_norm = float(residual.query.weight.grad.square().sum()) if residual.query.weight.grad is not None else 0.
            require(residual.bias.grad is not None and float(residual.bias.grad.abs().sum()) > 0., 'Bias has no task gradient')
            if arm == 'STATIC_RESIDUAL':
                require(all(p.grad is None for p in (*residual.query.parameters(), *residual.key.parameters())),
                        'Static inactive parameters received gradients')
            else:
                require(key_norm > 0., 'Residual key is a dead branch')
                require(query_norm == 0. if step == 0 else query_norm > 0., 'Unexpected staged query gradient')
            rows.append(dict(step=step+1, key_task_gradient_square=key_norm, query_task_gradient_square=query_norm))
            optimizer.step()
            synthetic_steps += 1
            require(train.base.tensor_tree_hash(split_state(model.state_dict())[0]) == source_hash, 'Optimizer changed frozen state')
        result['gradients'][arm] = rows
    model = models['RECEIVER_RESIDUAL']
    with torch.no_grad():
        model.selector.residual.key.weight.normal_(0., .25)
        model.selector.residual.bias.uniform_(-.2, .2)
        full = model(x)
        with model.intervention('BASE'):
            base = model(x)
            try:
                with model.intervention('STATIC_ONLY'):
                    raise RuntimeError('intentional restoration test')
            except RuntimeError:
                pass
            require(model.selector.override_mode == 'BASE', 'Nested intervention did not restore BASE')
        with model.intervention('QUERY_SELF'):
            self_query = model(x)
        require(model.selector.override_mode is None, 'Intervention leaked into subsequent forwards')
        require(torch.allclose(base, expected, atol=2e-6, rtol=2e-6), 'BASE did not disable the residual for the entire autoregressive forward')
        require(not torch.equal(full, self_query), 'Query evidence probe did not change predictions')
        require(not torch.equal(full, base), 'Nonzero residual has no output effect')
    result['interventions'] = dict(base_abs_max=float((base-expected).abs().max()),
        receiver_vs_self_abs_max=float((full-self_query).abs().max()), nested_exception_restored=True)
    # Finite difference on the key coordinate with the largest analytic signal.
    numerical = ReceiverResidualIGSTGNN(copy.deepcopy(anchor), 'RECEIVER_RESIDUAL').double()
    # The frozen native API stores predefined supports in a plain list.
    numerical.selector.native.pre_defined_graph = [graph.double()
        for graph in numerical.selector.native.pre_defined_graph]
    xd, yd = x.double(), target.double()
    loss = (numerical(xd)-yd).square().mean()
    loss.backward()
    parameter = numerical.selector.residual.key.weight
    index = int(parameter.grad.abs().argmax())
    analytic = float(parameter.grad.view(-1)[index])
    epsilon = 1e-5
    with torch.no_grad():
        parameter.view(-1)[index] += epsilon
        plus = float((numerical(xd)-yd).square().mean())
        parameter.view(-1)[index] -= 2*epsilon
        minus = float((numerical(xd)-yd).square().mean())
        parameter.view(-1)[index] += epsilon
    numeric = (plus-minus)/(2*epsilon)
    require(abs(analytic-numeric) <= 1e-7+1e-3*abs(analytic), 'GW4 finite-difference gradient differs')
    result['finite_difference'] = dict(analytic=analytic, numeric=numeric, epsilon=epsilon)
    # Round-trip optimizer/scheduler/RNG state, then compare the next update.
    def new_pair():
        candidate = ReceiverResidualIGSTGNN(copy.deepcopy(anchor), 'RECEIVER_RESIDUAL')
        opt = torch.optim.Adam([p for p in candidate.parameters() if p.requires_grad], lr=.002, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1, 2], gamma=.5)
        return candidate, opt, sched
    def step(candidate, opt, sched):
        candidate.train()
        opt.zero_grad(set_to_none=True)
        # Artificial stochastic training target tests restoration of RNG too.
        loss = (candidate(x)-target-.01*torch.randn_like(target)).square().mean()
        loss.backward()
        opt.step()
        sched.step()
    left, opt, sched = new_pair()
    step(left, opt, sched)
    synthetic_steps += 1
    snapshot = copy.deepcopy(dict(model=left.state_dict(), optimizer=opt.state_dict(),
        scheduler=sched.state_dict(), rng=train.base.rng_state()))
    with tempfile.TemporaryDirectory(prefix='gw4-synthetic-') as directory:
        path = Path(directory)/'resume.pt'
        torch.save(snapshot, path)
        step(left, opt, sched)
        synthetic_steps += 1
        expected_state = train.base.tensor_tree_hash((left.state_dict(), opt.state_dict(), sched.state_dict()))
        right, opt_right, sched_right = new_pair()
        restored = train.base.load_checkpoint(path)
        right.load_state_dict(restored['model'], strict=True)
        opt_right.load_state_dict(restored['optimizer'])
        sched_right.load_state_dict(restored['scheduler'])
        restore_rng(restored['rng'])
        step(right, opt_right, sched_right)
        synthetic_steps += 1
        require(expected_state == train.base.tensor_tree_hash((right.state_dict(), opt_right.state_dict(), sched_right.state_dict())),
                'Restored optimizer/scheduler/RNG produced a different next update')
    result['optimizer_resume'] = dict(next_update_tensors_exact=True,
        scope='synthetic_optimizer_scheduler_RNG_roundtrip_not_full_epoch_runner_interrupt_test')
    require(input_hash == train.base.tensor_tree_hash((x, target)), 'Verification mutated input or target')
    require(source_hash == train.base.tensor_tree_hash(anchor.state_dict()), 'Verification changed source fixture')
    result['synthetic_optimizer_updates'] = synthetic_steps
    return result


def native_backward(args):
    reference, source_identity, weights = certify(args.gw2_run,
        {key: getattr(args, 'gw2_'+key+'_eval') for key in ('best', 'last')})
    config = dict(source_identity['training_config'], device=args.device, bs=args.batch_size)
    modules = train.base.get_original_modules()
    require(modules.source_fingerprint == source_identity['source_fingerprint'], 'Native source differs')
    data = train.base.load_data(args.data_dir, config, modules,
        logging.getLogger('gw4-check'), check=False, training=True)
    from experiments.chronological.original_igstgnn_runtime import device_batch
    batch = device_batch(next(data.loaders['train_loader'].get_iterator()), args.device)
    x, y, incident, sensors = batch
    require(len(x) == args.batch_size, 'Requested batch unavailable; no smaller fallback')
    input_hash, result = train.base.tensor_tree_hash(batch), {}
    cuda = torch.device(args.device).type == 'cuda'
    for arm in ARMS:
        train.base.configure_runtime(args.device, args.threads, source_identity['training_config']['seed'])
        model = train.model_for_arm(data, modules, args.device, arm, weights)
        model.train()
        model.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        start = perf_counter()
        prediction = data.scaler.inverse_transform(model(x, incident_data=incident, sensor_data=sensors))
        target = data.scaler.inverse_transform(y)
        null = target.min() if target.min() < 1 else target.new_tensor(0.)
        loss = modules.metrics.masked_mae(prediction, target, null)
        loss.backward()
        if cuda:
            torch.cuda.synchronize(args.device)
        seconds = perf_counter()-start
        gradients = {}
        for name, parameter in model.named_parameters():
            if '.residual.' not in name:
                require(not parameter.requires_grad and parameter.grad is None, 'Native parameter received a gradient')
                continue
            if parameter.grad is not None:
                require(bool(torch.isfinite(parameter.grad).all()), 'Nonfinite native MAE gradient')
            gradients[name.rsplit('.residual.', 1)[1]] = dict(active=parameter.requires_grad,
                gradient_present=parameter.grad is not None,
                squared_norm=float(parameter.grad.double().square().sum()) if parameter.grad is not None else None)
        require(bool(torch.isfinite(loss)), 'Nonfinite native MAE')
        require(gradients['bias']['squared_norm'] > 0., 'Native MAE did not reach residual bias')
        if arm != 'STATIC_RESIDUAL':
            require(gradients['key.weight']['squared_norm'] > 0., 'Native MAE did not reach the new key projection')
        require(train.base.tensor_tree_hash(split_state(model.state_dict())[0]) == reference['source_model_tensor_sha256'],
                'Native engineering backward altered its anchor')
        result[arm] = dict(batch_size=len(x), nodes=x.shape[2], loss=float(loss.detach()),
            forward_backward_seconds=seconds,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(args.device)) if cuda else None,
            objective='native_inverse_transform_masked_mae_all_12_leads', gradients=gradients,
            real_data_optimizer_updates=0, new_candidate_forward_calls=1, baseline_forward_calls=0)
        print(json.dumps(dict(arm=arm, phase='native_backward_complete', **result[arm])), flush=True)
        del model, prediction, loss
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    require(input_hash == train.base.tensor_tree_hash(batch), 'Native backward changed its input')
    recertify(reference)
    return dict(reference=reference, requested_batch_size=args.batch_size, arms=result,
        smaller_batch_fallback=False, real_data_optimizer_updates=0,
        caveat='new_candidate_one_batch_engineering_not_baseline_validation_replay_or_accuracy_evidence')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--gw2-run', type=Path)
    parser.add_argument('--gw2-best-eval', type=Path)
    parser.add_argument('--gw2-last-eval', type=Path)
    parser.add_argument('--data-dir', type=Path, default=train.base.default_data_dir())
    parser.add_argument('--batch-size', type=int, default=48)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 3909:
        parser.error('batch-size must be in 1..3909')
    if args.output.exists():
        parser.error('Preserve verification evidence; use a fresh output filename')
    if not args.gw2_run and (args.gw2_best_eval or args.gw2_last_eval):
        parser.error('Evaluation paths require --gw2-run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = dict(schema='gw4_receiver_residual_verification_v1', status='GW4_VERIFICATION_RUNNING',
        engineering_only=True, design=train.DESIGN, baseline_new_training_updates=0,
        baseline_new_inference_calls=0, real_data_optimizer_updates=0, test_evaluated=False,
        execution_source={name: train.base.sha256(REPO/name) for name in (*train.EXECUTION_FILES,
            'experiments/chronological/verify_original_receiver_residual.py')})
    try:
        evidence['runtime'] = train.base.configure_runtime(args.device, args.threads)
        evidence['synthetic'] = synthetic_checks(args.device)
        gc.collect()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
        if args.gw2_run:
            evidence['native_backward'] = native_backward(args)
        evidence['status'] = 'GW4_VERIFICATION_PASS'
        evidence['full_native_backward_checked'] = bool(args.gw2_run)
        train.base.write_json(args.output, evidence)
        print(json.dumps(dict(status=evidence['status'], full_native_backward_checked=bool(args.gw2_run),
            output=str(args.output.resolve()))), flush=True)
        return evidence
    except BaseException as error:
        evidence.update(status='GW4_VERIFICATION_FAILED', error_type=type(error).__name__, error=str(error))
        train.base.write_json(args.output, evidence)
        raise


if __name__ == '__main__':
    main()
