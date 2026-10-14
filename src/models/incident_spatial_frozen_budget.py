"""Train only a zero-start location adapter on an immutable MIN checkpoint.

The backbone remains in evaluation mode while autograd follows its downstream
operations. Wrapping the complete forward in ``no_grad`` would remove the task
gradient needed to train the adapter, even though backbone weights are frozen.
"""
from src.models.incident_spatial import require
from src.models.incident_spatial_event_budget import EventBudgetIGSTGNN, from_minimal
from src.models.incident_spatial_residual import tensor_state_hash


ADAPTER_PREFIX = 'event_budget_adapter.'
FREEZE_VERSION = 'min_best_checkpoint_frozen_location_adapter_v1'


def _require_location_model(model):
    require(type(model) is EventBudgetIGSTGNN
            and model.event_budget_adapter.mode == 'location'
            and not hasattr(model.event_budget_adapter, 'semantic_encoder'),
            'Expected a location-only event budget model without a semantic encoder')


def frozen_state_hash(model):
    """Hash every retained MIN parameter and buffer, including the static graph."""
    _require_location_model(model)
    return tensor_state_hash({name: value for name, value in model.state_dict().items()
                              if not name.startswith(ADAPTER_PREFIX)})


def trainable_named_parameters(model):
    """Return the exact optimizer allowlist; reject any accidentally unfrozen MIN."""
    _require_location_model(model)
    result = []
    for name, parameter in model.named_parameters():
        is_adapter = name.startswith(ADAPTER_PREFIX)
        require(parameter.requires_grad == is_adapter,
                'Only the complete location adapter may require gradients: '+name)
        if is_adapter:
            result.append((name, parameter))
    require(result, 'Location adapter has no trainable parameters')
    return result


def set_frozen_train_mode(model):
    """Disable backbone dropout and running-stat updates; enable adapter training.

    Call this instead of ``model.train()`` before each training pass. Validation
    may use ``model.eval()`` normally. This function does not repair an unfrozen
    parameter or erase an unexpected gradient; the guard reports those errors.
    """
    _require_location_model(model)
    trainable_named_parameters(model)
    model.eval()
    model.event_budget_adapter.train()
    return model


def assert_frozen(model, expected_hash):
    """Fail if backbone flags, gradients, weights or buffers have changed."""
    _require_location_model(model)
    require(isinstance(expected_hash, str) and len(expected_hash) == 64,
            'Expected the frozen MIN SHA256 digest')
    trainable_named_parameters(model)
    for name, parameter in model.named_parameters():
        if not name.startswith(ADAPTER_PREFIX):
            require(parameter.grad is None, 'Frozen MIN has an unexpected gradient: '+name)
    for name, module in model.named_modules():
        if name != 'event_budget_adapter' and not name.startswith(ADAPTER_PREFIX):
            require(not module.training, 'Frozen MIN must remain in evaluation mode: '+(name or '<root>'))
    require(frozen_state_hash(model) == expected_hash, 'Frozen MIN parameters or buffers changed')
    return True


def build_frozen_location(base):
    """Copy a loaded MIN and return ``(location_model, freeze_certificate)``.

    The source model is untouched. The added adapter starts with an exactly zero
    final projection. Initial ON and every OFF forward equal a frozen-evaluation
    MIN reference at the same retained weights. Freezing can change floating
    point kernel dispatch compared with the original trainable-parameter MIN;
    callers must replay its saved metrics before freezing and record any such
    roundoff separately from the exact frozen-reference comparison.
    """
    model, conversion = from_minimal(base, 'location')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(ADAPTER_PREFIX))
        parameter.grad = None
    set_frozen_train_mode(model)
    digest = frozen_state_hash(model)
    require(digest == tensor_state_hash(base.state_dict()), 'Frozen copy differs from source MIN')
    assert_frozen(model, digest)
    trainable = trainable_named_parameters(model)
    certificate = dict(
        freeze_version=FREEZE_VERSION,
        conversion=conversion,
        frozen_state_sha256=digest,
        adapter_initial_sha256=tensor_state_hash(model.event_budget_adapter.state_dict()),
        frozen_parameters=sum(parameter.numel() for name, parameter in model.named_parameters()
                              if not name.startswith(ADAPTER_PREFIX)),
        trainable_parameters=sum(parameter.numel() for _, parameter in trainable),
        trainable_names=[name for name, _ in trainable],
        backbone_evaluation_mode=True,
        adapter_only_training_mode=True,
        backbone_gradients_disabled=True,
        downstream_autograd_required=True,
        semantic_encoder_present=False,
        optimizer_parameter_scope='event_budget_adapter_only',
        identity_reference='same_state_MIN_with_frozen_parameters_and_evaluation_mode',
        original_trainable_parameter_replay='fixed_tolerance_required_before_freezing',
        budget_scope='dimensionless_latent_exchange_budget',
    )
    return model, certificate
