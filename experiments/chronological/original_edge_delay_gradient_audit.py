"""Task/L2 gradient audit with no optimizer step; restore tensors, modes and RNG."""
import math
import random

import numpy as np
import torch

from experiments.chronological.original_igstgnn_runtime import device_batch, tensor_tree_hash
from experiments.chronological.train_original_igstgnn_baseline import restore_rng


def decay_only_adam(parameter=-2., updates=82, lr=.002, weight_decay=1e-5):
    """Scalar analytical counterfactual, zero task gradient (not grad=None)."""
    first, second = 0., 0.
    initial = parameter
    for step in range(1, updates + 1):
        gradient = weight_decay * parameter
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient * gradient
        parameter -= lr * (first / (1 - .9 ** step)) / (math.sqrt(second / (1 - .999 ** step)) + 1e-8)
    return dict(initial_logit=initial, final_logit=parameter, updates=updates,
        final_delay=2 / (1 + math.exp(-parameter)), task_gradient=0.,
        interpretation='analytical counterfactual; does not measure historical task gradients')


def group_name(name):
    if name.endswith('alignment_scale'):
        return 'alignment_scale'
    if name.endswith('edge_logits'):
        return 'legacy_edge_logits'
    if name.endswith('edge_deviation'):
        return 'edge_deviation'
    if '.delays.' in name:
        return 'state_scorer'
    return 'original_backbone'


def audit_gradients(model, batches, data, modules, *, weight_decay, clip_norm, device, phase):
    """Native full-horizon loss on supplied training batches; no loader shuffle.

Raw task gradients, globally clipped task gradients and Adam's coupled L2
term are recorded separately. L2 is inactive for grad=None, as in Adam.
Train-mode BN/dropout are used, then model state, existing gradients, modes
and all RNG streams are restored. No optimizer is constructed.
"""
    before = tensor_tree_hash(model.state_dict())
    saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    grads = {n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()}
    modes = {n: m.training for n, m in model.named_modules()}
    numpy_state = np.random.get_state()
    # A CPU audit must not initialize CUDA merely to capture its RNG.
    random_state = dict(torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]], python=random.getstate())
    rows, losses = [], []
    try:
        model.train()
        for batch_number, raw in enumerate(batches, 1):
            model.zero_grad(set_to_none=True)
            x, y, incident, sensors = device_batch(raw, device)
            prediction = model(x, y, incident_data=incident, sensor_data=sensors)
            prediction, target = data.scaler.inverse_transform(prediction), data.scaler.inverse_transform(y)
            mask = target.min() if target.min() < 1 else 0
            loss = modules.metrics.masked_mae(prediction, target, mask)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite audit loss')
            loss.backward()
            raw_norms = {n: None if p.grad is None else float(p.grad.double().norm())
                         for n, p in model.named_parameters()}
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
            losses.append(dict(batch=batch_number, loss=float(loss), raw_global_task_l2=float(total_norm)))
            for name, parameter in model.named_parameters():
                task = parameter.grad
                decay = weight_decay * parameter.detach().double()
                active = task is not None
                if active:
                    task = task.detach().double()
                    task_l2, decay_l2 = float(task.norm()), float(decay.norm())
                    dot = float((task * decay).sum())
                    combined_l2 = float((task + decay).norm())
                    cosine = dot / (task_l2 * decay_l2) if task_l2 and decay_l2 else None
                    nonzero = int(torch.count_nonzero(task))
                else:
                    task_l2 = decay_l2 = dot = combined_l2 = 0.
                    cosine, nonzero = None, 0
                rows.append(dict(batch=batch_number, parameter=name, group=group_name(name),
                    elements=parameter.numel(), gradient_present=active, task_nonzero_elements=nonzero,
                    raw_task_l2=raw_norms[name], clipped_task_l2=task_l2,
                    active_decay_l2=decay_l2, potential_decay_l2=float(decay.norm()),
                    task_decay_dot=dot, task_decay_cosine=cosine, combined_l2=combined_l2,
                    decay_to_task_l2=decay_l2 / task_l2 if task_l2 else None))
    finally:
        model.load_state_dict(saved, strict=True)
        for name, parameter in model.named_parameters():
            parameter.grad = grads[name]
        for name, module in model.named_modules():
            module.training = modes[name]
        restore_rng(random_state)
    if before != tensor_tree_hash(model.state_dict()):
        raise ValueError('Gradient audit failed to restore model tensors')
    groups = []
    for batch in range(1, len(losses) + 1):
        for group in sorted({r['group'] for r in rows}):
            selected = [r for r in rows if r['batch'] == batch and r['group'] == group]
            task = math.sqrt(sum(r['clipped_task_l2'] ** 2 for r in selected))
            decay = math.sqrt(sum(r['active_decay_l2'] ** 2 for r in selected))
            groups.append(dict(batch=batch, group=group, elements=sum(r['elements'] for r in selected),
                gradient_none_tensors=sum(not r['gradient_present'] for r in selected),
                task_nonzero_elements=sum(r['task_nonzero_elements'] for r in selected),
                clipped_task_l2=task, active_decay_l2=decay,
                decay_to_task_l2=decay / task if task else None))
    return dict(schema='original_delay_task_decay_gradient_audit_v1', phase=phase, losses=losses,
        groups=groups, parameters=rows, optimizer_updates=0, backward_calls=len(losses),
        model_restored=True, rng_restored=True, full_horizon=12,
        caveat='gradient norm ratios are not causal shares of historical updates; initial v2 lag gradient is zero until residual scales move')
