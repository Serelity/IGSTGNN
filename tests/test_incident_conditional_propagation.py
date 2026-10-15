"""CPU algebra/interface tests; no data or predictive-performance claims."""
from copy import deepcopy
import math
import unittest

import torch

from src.models.igstgnn import STLocalizedConv
from src.models.incident_conditional_propagation import ConditionalMessagePropagation


def localized(graph):
    return graph.unsqueeze(-2).expand(*graph.shape[:-1], 3, graph.shape[-1]).reshape(*graph.shape[:-1], 3*graph.shape[-1])


def fixture(n=4, c=8, dtype=torch.float64, checkpoint_messages=False, chunk=2):
    adjacency = torch.zeros(n, n, dtype=dtype)
    for i in range(n-1):
        adjacency[i, i+1] = 1.
    adjacency[-1, 0] = .5
    pre = [adjacency/adjacency.sum(-1, keepdim=True).clamp_min(1),
           adjacency.T/adjacency.T.sum(-1, keepdim=True).clamp_min(1)]
    old = STLocalizedConv(c, pre_defined_graph=pre, k_s=2, k_t=3,
                         dropout=0., use_pre=True, dy_graph=True, sta_graph=True).to(dtype=dtype)
    module = ConditionalMessagePropagation(old, adjacency, receiver_chunk_size=chunk,
                                          checkpoint_messages=checkpoint_messages).to(dtype=dtype)
    module.eval()
    x = torch.randn(2, 5, n, c, dtype=dtype)
    dyn = []
    for _ in range(2):
        base = torch.rand(2, n, n, dtype=dtype)
        for graph in (base, base @ base):
            dyn.append(localized(graph))
    static = [torch.rand(n, n, dtype=dtype)]
    report = dict(condition=torch.randn(2, n, c, dtype=dtype),
                  support=torch.tensor([[True]+[False]*(n-1), [False]*(n-1)+[True]]),
                  available=torch.ones(2, dtype=torch.bool))
    return module, adjacency, x, dyn, static, report


def activate(module):
    with torch.no_grad():
        module.event_logit.weight.normal_(0., .2)
        module.event_logit.bias.fill_(.1)
        module.event_value.weight.normal_(0., .2)
        module.event_value.bias.fill_(.05)


def dense_reference(module, adjacency, x, dynamic, static, report=None,
                    kind=None, block=None, logit=True, value=True):
    """Direct node/position summation, independent of sparse/chunk helpers."""
    b, t, n, c = x.shape
    q = t-2
    history = torch.zeros(t, dtype=torch.long, device=x.device)
    kind = history if kind is None else kind
    block = history-1 if block is None else block
    masks = []
    identity = torch.eye(n, dtype=torch.bool, device=x.device)
    for direction in (adjacency > 0, adjacency.T > 0):
        masks.extend((direction & ~identity,
                      ((direction.to(torch.int64) @ direction.to(torch.int64)) > 0) & ~identity))
    banks = [identity] + masks + masks + [masks[0] | masks[2], masks[1] | masks[3]]
    priors = [torch.eye(n, dtype=x.dtype, device=x.device)]
    priors += [getattr(module, 'predefined_{}'.format(i)).reshape(n, 3, n)[:, 0] for i in range(4)]
    priors += [g.reshape(b, n, 3, n)[:, :, 0] for g in dynamic]
    priors += [static[0], static[0] @ static[0]]
    output = []
    for batch in range(b):
        windows = []
        for window in range(q):
            receivers = []
            p = torch.cat((torch.eye(3, dtype=x.dtype, device=x.device),
                           module.kind_embedding(kind[window:window+3]),
                           module.block_embedding(block[window:window+3]+1)), -1)
            for i in range(n):
                bank_results = []
                u = torch.tanh(module.query(x[batch, window+2, i]))
                for bank, mask in enumerate(banks):
                    graph = priors[bank][batch] if priors[bank].ndim == 3 else priors[bank]
                    graph = graph * mask.to(x.dtype)
                    scores, messages = [], []
                    for j in range(n):
                        if not bool(graph[i, j] > 0):
                            continue
                        gamma = report is not None and bool(report['available'][batch]) and bool(report['support'][batch, i] | report['support'][batch, j])
                        r = (torch.tanh(module.relation(torch.cat((report['condition'][batch, i], report['condition'][batch, j]))))
                             if gamma else None)
                        for ell in range(3):
                            source = x[batch, window+ell, j]
                            z = torch.tanh(module.key(source)+module.position(p[ell]))
                            score = (u*z).sum()/math.sqrt(c)+torch.log(graph[i, j])
                            message = module.source_value[ell](source)
                            if gamma:
                                interaction = u*z*torch.tanh(module.time_factor(p[ell]))*r
                                if logit:
                                    score = score+module.beta*torch.tanh(module.event_logit(interaction)).squeeze(-1)
                                if value:
                                    message = message+module.eta*torch.tanh(module.event_value(torch.cat((interaction, r))))
                            scores.append(score)
                            messages.append(message)
                    if not scores:
                        bank_results.append(x.new_zeros(c))
                    else:
                        rho = 1. if bank == 0 else 3*graph[i].sum()
                        bank_results.append(rho*(torch.softmax(torch.stack(scores), 0)[:, None]*torch.stack(messages)).sum(0))
                receivers.append(module.gcn_updt(torch.cat(bank_results)))
            windows.append(torch.stack(receivers))
        output.append(torch.stack(windows))
    return module.dropout(torch.stack(output))


class ConditionalPropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(113)
        self.module, self.adjacency, self.x, self.dynamic, self.static, self.report = fixture()

    def test_dense_reference_and_chunk_independence(self):
        activate(self.module)
        result = self.module(self.x, self.dynamic, self.static, **self.report)
        reference = dense_reference(self.module, self.adjacency, self.x,
                                    self.dynamic, self.static, self.report)
        torch.testing.assert_close(result, reference, atol=2e-13, rtol=2e-13)
        self.module.receiver_chunk_size = 1
        torch.testing.assert_close(self.module(self.x, self.dynamic, self.static, **self.report), reference,
                                   atol=2e-13, rtol=2e-13)
        self.assertEqual(result.shape, (2, 3, 4, 8))

    def test_dense_reference_gradients_parameters_input_graph_report(self):
        activate(self.module)
        self.x.requires_grad_()
        self.report['condition'].requires_grad_()
        self.dynamic = [g.detach().requires_grad_() for g in self.dynamic]
        self.static = [g.detach().requires_grad_() for g in self.static]
        inputs = [self.x, self.report['condition']] + self.dynamic + self.static + list(self.module.parameters())
        result = self.module(self.x, self.dynamic, self.static, **self.report)
        reference = dense_reference(self.module, self.adjacency, self.x,
                                    self.dynamic, self.static, self.report)
        target = torch.randn_like(result)
        actual_grads = torch.autograd.grad((result*target).sum(), inputs, retain_graph=True)
        expected_grads = torch.autograd.grad((reference*target).sum(), inputs)
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)

    def test_zero_initialization_and_first_step_readout_gradients(self):
        event = self.module(self.x, self.dynamic, self.static, **self.report)
        history = self.module(self.x, self.dynamic, self.static, report_enabled=False)
        self.assertTrue(torch.equal(event, history))
        event.square().sum().backward()
        for name in ('event_logit', 'event_value'):
            self.assertGreater(float(getattr(self.module, name).weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.equal(self.module.relation.weight.grad,
                                    torch.zeros_like(self.module.relation.weight.grad)))

    def test_report_off_ignores_inputs_and_parameters(self):
        activate(self.module)
        history = self.module(self.x, self.dynamic, self.static)
        off = self.module(self.x, self.dynamic, self.static, report_enabled=False,
                          condition='ignored', support='ignored', available='ignored')
        self.assertTrue(torch.equal(history, off))
        off.sum().backward()
        self.assertIsNone(self.module.relation.weight.grad)
        self.assertIsNone(self.module.event_logit.weight.grad)
        self.assertIsNone(self.module.event_value.weight.grad)

    def test_unavailable_zero_support_and_bias_gate_exactly_revert(self):
        activate(self.module)
        history = self.module(self.x, self.dynamic, self.static)
        for field in ('available', 'support'):
            report = dict(self.report)
            report[field] = torch.zeros_like(report[field])
            candidate = self.module(self.x, self.dynamic, self.static, **report)
            self.assertTrue(torch.equal(candidate, history), field)

    def test_partial_availability_and_support_gates(self):
        activate(self.module)
        report = dict(self.report, available=torch.tensor([True, False]))
        event = self.module(self.x, self.dynamic, self.static, **report)
        history = self.module(self.x, self.dynamic, self.static)
        self.assertTrue(torch.equal(event[1], history[1]))
        self.assertFalse(torch.equal(event[0], history[0]))

    def test_fixed_support_report_content_changes_homogeneous_and_zero_values(self):
        activate(self.module)
        with torch.no_grad():
            for layer in self.module.source_value:
                layer.weight.zero_()
                layer.bias.zero_()
        changed = dict(self.report, condition=self.report['condition']+1.)
        first = self.module(torch.zeros_like(self.x), self.dynamic, self.static, **self.report)
        second = self.module(torch.zeros_like(self.x), self.dynamic, self.static, **changed)
        self.assertFalse(torch.equal(first, second))
        with torch.no_grad():
            for layer in self.module.source_value:
                layer.bias.fill_(.2)
        first = self.module(self.x, self.dynamic, self.static, **self.report)
        second = self.module(self.x, self.dynamic, self.static, **changed)
        self.assertFalse(torch.equal(first, second))

    def test_logit_only_has_homogeneous_blind_spot_value_only_does_not(self):
        activate(self.module)
        with torch.no_grad():
            for layer in self.module.source_value:
                layer.weight.zero_()
                layer.bias.fill_(.2)
        report = dict(self.report, condition=self.report['condition']+1.)
        first = self.module(self.x, self.dynamic, self.static, **self.report, value_enabled=False)
        second = self.module(self.x, self.dynamic, self.static, **report, value_enabled=False)
        torch.testing.assert_close(first, second, atol=2e-15, rtol=2e-15)
        first = self.module(self.x, self.dynamic, self.static, **self.report, logit_enabled=False)
        second = self.module(self.x, self.dynamic, self.static, **report, logit_enabled=False)
        self.assertFalse(torch.equal(first, second))

    def test_joint_interaction_cross_difference_is_nonzero(self):
        activate(self.module)
        h = self.x[0, :3]
        p = torch.cat((torch.eye(3, dtype=h.dtype),
                       self.module.kind_embedding(torch.zeros(3, dtype=torch.long)),
                       self.module.block_embedding(torch.zeros(3, dtype=torch.long))), -1)
        u = torch.tanh(self.module.query(h[-1, 0]))
        scores = []
        for j in (1, 2):
            r = torch.tanh(self.module.relation(torch.cat((self.report['condition'][0, 0], self.report['condition'][0, j]))))
            row = []
            for ell in (0, 1):
                z = torch.tanh(self.module.key(h[ell, j])+self.module.position(p[ell]))
                interaction = u*z*torch.tanh(self.module.time_factor(p[ell]))*r
                row.append(torch.tanh(self.module.event_logit(interaction)).squeeze())
            scores.append(row)
        difference = scores[0][0]-scores[0][1]-scores[1][0]+scores[1][1]
        self.assertGreater(abs(float(difference.detach())), 1e-8)

    def test_hard_edges_disconnected_report_change_cannot_cross(self):
        adjacency = self.adjacency.clone()
        adjacency.zero_()
        adjacency[0, 1] = adjacency[2, 3] = 1.
        old = STLocalizedConv(8, pre_defined_graph=[adjacency, adjacency.T],
                             k_s=2, k_t=3, dropout=0., use_pre=True).double()
        module = ConditionalMessagePropagation(old, adjacency).double().eval()
        activate(module)
        report = dict(self.report, support=torch.tensor([[True, False, False, False]]).expand(2, -1))
        changed = dict(report, condition=report['condition'].clone())
        changed['condition'][:, :2] += 1.
        first = module(self.x, self.dynamic, self.static, **report)
        second = module(self.x, self.dynamic, self.static, **changed)
        self.assertTrue(torch.equal(first[:, :, 2:], second[:, :, 2:]))
        self.assertFalse(torch.equal(first[:, :, :2], second[:, :, :2]))

    def test_empty_graph_banks_are_finite_and_local_self_survives(self):
        adjacency = torch.zeros_like(self.adjacency)
        old = STLocalizedConv(8, pre_defined_graph=[adjacency, adjacency],
                             k_s=2, k_t=3, dropout=0., use_pre=True).double()
        module = ConditionalMessagePropagation(old, adjacency).double().eval()
        activate(module)
        result = module(self.x, self.dynamic, self.static, **self.report)
        reference = dense_reference(module, adjacency, self.x, self.dynamic, self.static, self.report)
        torch.testing.assert_close(result, reference)
        self.assertTrue(bool(torch.isfinite(result).all()))
        result.sum().backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters()))

    def test_node_relabeling_equivariance(self):
        activate(self.module)
        order = torch.tensor([2, 0, 3, 1])
        adjacency = self.adjacency[order][:, order]
        pre = [getattr(self.module, 'predefined_{}'.format(i)).reshape(4, 3, 4)[:, 0]
               for i in (0, 2)]
        old = STLocalizedConv(8, pre_defined_graph=[g[order][:, order] for g in pre],
                             k_s=2, k_t=3, dropout=0., use_pre=True).double()
        permuted = ConditionalMessagePropagation(old, adjacency).double().eval()
        with torch.no_grad():
            for name, value in self.module.named_parameters():
                dict(permuted.named_parameters())[name].copy_(value)
        dyn = [localized(g.reshape(2, 4, 3, 4)[:, :, 0][:, order][:, :, order]) for g in self.dynamic]
        static = [self.static[0][order][:, order]]
        report = dict(self.report, condition=self.report['condition'][:, order], support=self.report['support'][:, order])
        result = permuted(self.x[:, :, order], dyn, static, **report)
        expected = self.module(self.x, self.dynamic, self.static, **self.report)[:, :, order]
        torch.testing.assert_close(result, expected, atol=3e-13, rtol=3e-13)

    def test_token_history_generated_metadata_and_single_window(self):
        activate(self.module)
        x = self.x[:, :3]
        for kind, block in (([0, 0, 1], [-1, -1, 0]), ([0, 1, 1], [-1, 0, 1]),
                            ([1, 1, 1], [0, 1, 2])):
            kind, block = torch.tensor(kind), torch.tensor(block)
            result = self.module(x, self.dynamic, self.static, **self.report,
                                 token_kind=kind, token_block=block)
            expected = dense_reference(self.module, self.adjacency, x, self.dynamic,
                                       self.static, self.report, kind, block)
            torch.testing.assert_close(result, expected)
            self.assertEqual(result.shape, (2, 1, 4, 8))

    def test_dtype_shape_bool_and_finite_contracts(self):
        bad_reports = [dict(self.report, support=self.report['support'].float()),
                       dict(self.report, available=self.report['available'].long()),
                       dict(self.report, condition=self.report['condition'].float()),
                       dict(self.report, condition=self.report['condition'][..., :2]),
                       dict(self.report, support=None),
                       dict(self.report, condition=torch.full_like(self.report['condition'], float('nan')))]
        for report in bad_reports:
            with self.assertRaises(ValueError):
                self.module(self.x, self.dynamic, self.static, **report)
        for x in (self.x.float(), self.x[:, :2], self.x[:, :, :3], self.x[:0], self.x*float('nan')):
            with self.assertRaises(ValueError):
                self.module(x, self.dynamic, self.static)
        for name in ('report_enabled', 'logit_enabled', 'value_enabled', 'collect'):
            with self.assertRaises(TypeError):
                self.module(self.x, self.dynamic, self.static, **{name: 1})
        with self.assertRaises(ValueError):
            self.module(self.x, self.dynamic, self.static, token_kind=torch.zeros(5), token_block=torch.full((5,), -1))
        with self.assertRaises(ValueError):
            self.module(self.x, self.dynamic, self.static, token_kind=torch.ones(5, dtype=torch.long), token_block=torch.full((5,), -1))

    def test_graph_validation_zero_prior_edges_do_not_reopen(self):
        dynamic = [g.clone() for g in self.dynamic]
        dynamic[0][0, 0, 1] = -1.
        with self.assertRaises(ValueError):
            self.module(self.x, dynamic, self.static)
        dynamic = [g.clone() for g in self.dynamic]
        dynamic[0][0, 0, 1] += .1
        with self.assertRaises(ValueError):
            self.module(self.x, dynamic, self.static)
        with self.assertRaises(ValueError):
            self.module(self.x, self.dynamic[:3], self.static)
        zeros = [torch.zeros_like(g) for g in self.dynamic]
        result = self.module(self.x, zeros, [torch.zeros_like(self.static[0])])
        expected = dense_reference(self.module, self.adjacency, self.x, zeros,
                                   [torch.zeros_like(self.static[0])])
        torch.testing.assert_close(result, expected)

    def test_compatible_weights_only_and_serialization(self):
        names = set(self.module.state_dict())
        self.assertFalse(any('fc_list_updt' in name or 'incident_exchange' in name or 'bn.' in name for name in names))
        clone = deepcopy(self.module)
        clone.load_state_dict(self.module.state_dict(), strict=True)
        self.assertTrue(torch.equal(clone(self.x, self.dynamic, self.static, **self.report),
                                    self.module(self.x, self.dynamic, self.static, **self.report)))

    def test_second_step_factor_and_condition_gradients(self):
        self.report['condition'].requires_grad_()
        optimizer = torch.optim.SGD(self.module.parameters(), lr=.01)
        target = torch.randn(2, 3, 4, 8, dtype=self.x.dtype)
        first = self.module(self.x, self.dynamic, self.static, **self.report)
        (first-target).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.report['condition'].grad = None
        second = self.module(self.x, self.dynamic, self.static, **self.report)
        (second-target).square().mean().backward()
        self.assertGreater(float(self.module.relation.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.report['condition'].grad.abs().sum()), 0.)

    def test_training_checkpoint_gradients_match_direct_and_no_stale_stats(self):
        activate(self.module)
        direct = self.module.train()
        recompute = deepcopy(direct)
        recompute.checkpoint_messages = True
        x1 = self.x.clone().requires_grad_()
        x2 = self.x.clone().requires_grad_()
        y1 = direct(x1, self.dynamic, self.static, **self.report, collect=True)
        y2 = recompute(x2, self.dynamic, self.static, **self.report, collect=True)
        torch.testing.assert_close(y1, y2)
        y1.square().sum().backward()
        y2.square().sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad)
        for p1, p2 in zip(direct.parameters(), recompute.parameters()):
            torch.testing.assert_close(p1.grad, p2.grad)
        self.assertTrue(recompute.last_details['checkpoint_messages'])
        self.assertLessEqual(recompute.last_details['statistics']['event_logit_abs_max'], 1.)
        self.assertLessEqual(recompute.last_details['statistics']['event_value_abs_max'], 1.)
        self.assertTrue(all(isinstance(value, torch.Tensor) and value.ndim == 0 and not value.requires_grad
                            for value in recompute.last_details['statistics'].values()))
        self.assertEqual(recompute.last_details['statistics']['active_report_edges'].dtype, torch.int64)
        recompute(x2, self.dynamic, self.static)
        self.assertIsNone(recompute.last_details)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_device_contract_and_float32_cuda(self):
        module, _, x, dynamic, static, report = fixture(dtype=torch.float32)
        module.cuda()
        x, dynamic, static = x.cuda(), [g.cuda() for g in dynamic], [g.cuda() for g in static]
        report = {key: value.cuda() for key, value in report.items()}
        result = module(x, dynamic, static, **report)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(result).all()))
        report['support'] = report['support'].cpu()
        with self.assertRaises(ValueError):
            module(x, dynamic, static, **report)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_autocast_checkpoint_finite_gradients(self):
        module, _, x, dynamic, static, report = fixture(dtype=torch.float32,
                                                       checkpoint_messages=True)
        activate(module)
        module.cuda().train()
        x = x.cuda().requires_grad_()
        dynamic, static = [g.cuda() for g in dynamic], [g.cuda() for g in static]
        report = {key: value.cuda() for key, value in report.items()}
        with torch.autocast('cuda', dtype=torch.float16):
            result = module(x, dynamic, static, **report, collect=True)
            loss = result.float().square().mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


if __name__ == '__main__':
    unittest.main()
