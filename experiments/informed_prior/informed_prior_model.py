"""
informed_prior_model.py - MAP logistic regression with a population-
informed prior mean, as a lightweight stand-in for a full Bayesian
treatment.

Model: minimize, over intercept and beta,
    logistic_loss(intercept, beta; X, y) + lambda * sum((beta_j - prior_j)^2)

This is the MAP estimate under independent Gaussian priors N(prior_j,
sigma_j^2) on each coefficient (lambda = 1/(2*sigma^2), one shared lambda
across features rather than per-feature variances). prior=0 recovers
standard ridge-toward-zero logistic regression - used below as an internal
correctness check against sklearn's own L2 fit.

Fit via scipy.optimize.minimize (L-BFGS-B) with an analytic gradient - the
objective is convex, so this converges to the same MAP estimate a full
PyMC posterior's mode would, without needing that heavier dependency.

Inherits sklearn.base.BaseEstimator so it can be used anywhere a fitted
sklearn classifier is expected (get_params/set_params come for free from
the __init__ signature) - needed so it can drop into the same clone()
machinery host/training/cross_validation.py uses.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _neg_log_posterior(params: np.ndarray, X: np.ndarray, y: np.ndarray, prior: np.ndarray, lam: float) -> float:
    intercept, beta = params[0], params[1:]
    p = _sigmoid(intercept + X @ beta)
    eps = 1e-12
    nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return nll + lam * np.sum((beta - prior) ** 2)


def _grad(params: np.ndarray, X: np.ndarray, y: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    intercept, beta = params[0], params[1:]
    resid = _sigmoid(intercept + X @ beta) - y
    grad_intercept = np.sum(resid)
    grad_beta = X.T @ resid + 2 * lam * (beta - prior)
    return np.concatenate([[grad_intercept], grad_beta])


class InformedPriorLogisticRegression(BaseEstimator, ClassifierMixin):
    """MAP logistic regression shrinking coefficients toward `prior`
    (not zero), at strength `lam`. lam=0 -> unpenalized MLE on this data
    alone. Large lam -> coefficients stay close to `prior` regardless of
    what this fold's data says.
    """

    def __init__(self, prior=None, lam: float = 1.0):
        self.prior = prior
        self.lam = lam

    def fit(self, X: np.ndarray, y: np.ndarray) -> "InformedPriorLogisticRegression":
        prior = np.zeros(X.shape[1]) if self.prior is None else np.asarray(self.prior, dtype=float)
        if len(prior) != X.shape[1]:
            raise ValueError(f"prior length {len(prior)} != n_features {X.shape[1]}")

        x0 = np.zeros(X.shape[1] + 1)
        x0[1:] = prior  # warm-start at the prior
        res = minimize(_neg_log_posterior, x0, args=(X, y, prior, self.lam), jac=_grad, method="L-BFGS-B")

        self.intercept_ = np.array([res.x[0]])
        self.coef_ = res.x[1:].reshape(1, -1)
        self.converged_ = res.success
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = _sigmoid(self.intercept_[0] + X @ self.coef_[0])
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _self_check():
    """prior=0 should reproduce a plain L2-penalized logistic regression:
    sklearn parameterizes as C = 1/(2*lam*n) roughly (up to how sklearn
    scales its penalty by n_samples internally) - rather than chase an
    exact C<->lam identity, just confirm both fits land in the same
    ballpark and predict similarly, as a sanity check on the optimizer.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n, p = 200, 10
    X = rng.normal(size=(n, p))
    true_beta = rng.normal(size=p)
    y = (rng.uniform(size=n) < _sigmoid(X @ true_beta)).astype(int)

    lam = 1.0
    ours = InformedPriorLogisticRegression(prior=np.zeros(p), lam=lam).fit(X, y)
    sk = LogisticRegression(C=1.0 / (2 * lam), max_iter=2000).fit(X, y)

    corr = np.corrcoef(ours.coef_[0], sk.coef_[0])[0, 1]
    print(f"Self-check (prior=0 vs sklearn L2, informal): coefficient correlation = {corr:.4f}"
          f"  ({'looks consistent' if corr > 0.9 else 'CHECK - unexpectedly low agreement'})")


if __name__ == "__main__":
    _self_check()
