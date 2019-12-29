import logging

from autograd import grad
from autograd.misc.optimizers import adam


def minimize(loss, init_params, step_size=0.05, num_iters=10000, optimizer=adam):
    """TODO: we should be doing early stopping on a hold-out set"""
    min_loss, min_params, min_iter = None, None, None

    def at_iter(params, iter, gradient):
        nonlocal min_loss, min_params, min_iter
        l = loss(params, iter)
        if min_loss is not None and l >= min_loss:
            return
        min_loss = l
        min_params = params
        min_iter = iter

    optimizer(
        grad(loss),
        init_params,
        step_size=step_size,
        num_iters=num_iters,
        callback=at_iter,
    )
    logging.info(f"minimum loss {min_loss} at iter {min_iter} of {num_iters}")

    return min_params, min_loss
