"""Backtrack an Adam candidate against fixed teacher-action budgets."""

import copy

import torch


def budgeted_step(policy, optimizer, measure, budget, *, backtracks=5):
    """Keep accepted momentum; restore parameters AND Adam state on total rejection.

    ``measure`` returns one action MSE per reference group. The constraint only
    covers these fixed histories; simulation regression checks remain necessary.
    """
    parameters = list(policy.parameters())
    original = [p.detach().clone() for p in parameters]
    state = {p: copy.deepcopy(optimizer.state[p]) for p in parameters if p in optimizer.state}
    optimizer.step()
    delta = [p.detach() - old for p, old in zip(parameters, original, strict=True)]
    accepted, measured = 0.0, None
    with torch.no_grad():
        for attempt in range(backtracks + 1):
            scale = 0.5**attempt
            if attempt:
                for p, old, change in zip(parameters, original, delta, strict=True):
                    p.copy_(old + scale * change)
            measured = measure()
            if torch.isfinite(measured).all() and (measured <= budget**2 + 1e-10).all():
                accepted = scale
                break
        if not accepted:
            for p, old in zip(parameters, original, strict=True):
                p.copy_(old)
            optimizer.state.clear()
            optimizer.state.update(state)
            measured = measure()
    if not torch.isfinite(measured).all():
        raise FloatingPointError("Nonfinite teacher-action guard measurement")
    return measured, accepted
