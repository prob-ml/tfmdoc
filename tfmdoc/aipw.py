import torch


def aipw_estimator(y, t, pi_x, ey_x):
    """Estimate average treatment effect
    using the augmented inverse propensity-weighted estimator.

    Args:
        y (tensor): (N, ) array of outcomes
        t (tensor): (N,) array of treatments
        pi_x (tensor): (N,) array of propensity scores
        ey_x (tensor): (N, 2) array of estimated conditional outcomes.
            First column, corresponding to control group,
            is conditioned on no treatment assigned: E[Y | T = 0, X].
            Second column, corresponding to treatment group,
            is conditioned on treatment assigned: E[Y | T = 1, X].

    """  # noqa: RST301
    # calculate inverse propensity weighted estimator
    ipw = ((t * y) / pi_x) - (((1 - t) * y) / (1 - pi_x))
    # calculate regression adjustment
    # a weighted average of the two regression estimators
    regr = (
        (t - pi_x) / (pi_x * (1 - pi_x)) * ((1 - pi_x) * ey_x[:, 1] + pi_x * ey_x[:, 0])
    )

    return torch.mean(ipw - regr)
