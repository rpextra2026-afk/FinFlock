"""Ownership game: identities on reduced markets, sleeve, conditional rule, path, anchor, volume.

Mirrors spec 7.8 #1 and 12.2 #9. Reduced markets: N <= 30 assets, K <= 4 types, random covariance.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from oryxis.mfg.path import (
    anchor_outside_belief,
    anchored_path,
    expected_path,
    predicted_increments,
    price_path,
)
from oryxis.mfg.precompute import (
    FundType,
    GameError,
    amihud_liquidity,
    assemble_month_game,
    baseline_cost_scale,
    estimate_covariance,
    min_variance_weights,
    projected_precision,
    rebalancing_matrix,
)
from oryxis.mfg.push import conditional_prices, end_point, static_push_regressors
from oryxis.mfg.speed import EventKnobs, event_knobs, stable_root
from oryxis.mfg.validate import exact_prices
from oryxis.mfg.volume import covariance_shift, event_volume, fund_trades

BETA_DAILY = 0.9996


def random_market(seed=1, n=8, k=3, sleeve=False, equicorr=False, theta_zero=False):
    rng = np.random.default_rng(seed)
    size = n + int(sleeve)
    if equicorr:
        sigma = 0.04 / 250 * (0.7 * np.eye(size) + 0.3 * np.ones((size, size)))
    else:
        factors = rng.normal(size=(size, 2)) * 0.12
        sigma = (factors @ factors.T + np.diag(rng.uniform(0.01, 0.04, size))) / 250
    universes = [sorted(rng.choice(n, rng.integers(3, n + 1), replace=False).tolist()) for _ in range(k)]
    universes[0] = list(range(n))
    x_plus = []
    for universe in universes:
        w = np.zeros(size)
        w[universe] = rng.uniform(0.5, 1.5, len(universe))
        if sleeve:
            w = w / w.sum() * (1 - rng.uniform(0.1, 0.5))
            w[n] = 1 - w[:n].sum()
        else:
            w = w / w.sum()
        x_plus.append(w)
    return dict(
        N=n, K=k, sleeve=sleeve, Sig=sigma, Us=universes, x=x_plus,
        a=rng.uniform(0.5, 2.0, k), gam=rng.uniform(2.0, 6.0, k), rho=rng.uniform(0.02, 0.2, k),
        theta=np.zeros(k) if theta_zero else rng.uniform(0.0, 1.0, k), vth=np.zeros(k),
        beta=BETA_DAILY, ell=rng.uniform(0.5, 1.5, n), lam=rng.uniform(1.0, 4.0, k),
    )


def build_game(mk, psi_scale=1.0):
    """MonthGame whose liquidity is ``ell / psi_scale`` (impact scale ``psi_scale``)."""
    n, size = mk["N"], mk["N"] + int(mk["sleeve"])
    types = []
    for k in range(mk["K"]):
        x = mk["x"][k]
        universe = np.flatnonzero(x > 0)
        S_plus = projected_precision(mk["Sig"], universe)
        R_plus = rebalancing_matrix(x)
        x_mv = min_variance_weights(mk["Sig"], universe)
        beta_k = mk["beta"] * (1 - mk["rho"][k])
        types.append(
            FundType(
                name=f"t{k}", group="x", book=mk["a"][k], sp_cover=float(x[:n].sum()),
                x_plus=x, x_sp=x[:n], x_mv_sp=x_mv[:n], S=S_plus[:n, :n], R=R_plus[:n, :n],
                S_sleeve=S_plus[n, :n] if mk["sleeve"] else np.zeros(n),
                R_sleeve=R_plus[n, :n] if mk["sleeve"] else np.zeros(n),
                x_sleeve=float(x[n]) if mk["sleeve"] else 0.0,
                x_mv_sleeve=float(x_mv[n]) if mk["sleeve"] else 0.0,
                gamma_bar=mk["gam"][k], rho=mk["rho"][k], theta=mk["theta"][k], vartheta=mk["vth"][k],
                mandate=0.5, trade_rate_bar=0.0, cost_bar=mk["lam"][k], beta_k=beta_k,
                omega_bar=mk["rho"][k] / mk["gam"][k],
            )
        )
    sigma_full = mk["Sig"] if mk["sleeve"] else np.pad(mk["Sig"], ((0, 1), (0, 1)))
    return assemble_month_game([f"a{i}" for i in range(n)], sigma_full, mk["ell"] / psi_scale, types, mk["sleeve"])


def neutral_knobs(game, cfg):
    k = len(game.types)
    zeros = np.zeros(k)
    return event_knobs(game, cfg, fear_minus_anger=zeros, arousal=zeros, state_z=zeros, trait_b=zeros)


# --- Lemma P and the building blocks ---------------------------------------------------------


def test_lemma_p_both_forms_agree_and_s_is_a_laplacian():
    mk = random_market(seed=2)
    for universe in mk["Us"]:
        S = projected_precision(mk["Sig"], np.array(universe))
        basis = np.zeros((mk["N"], len(universe)))
        basis[universe, range(len(universe))] = 1.0
        V = basis @ sla.null_space(np.ones((1, len(universe))))
        other = V @ np.linalg.inv(V.T @ mk["Sig"] @ V) @ V.T
        assert np.allclose(S, other, atol=1e-9 * np.abs(S).max())
        assert np.allclose(S, S.T) and np.abs(S.sum(axis=0)).max() < 1e-8 * np.abs(S).max()
        outside = np.setdiff1d(np.arange(mk["N"]), universe)
        assert np.allclose(S[np.ix_(outside, range(mk["N"]))], 0)
        assert np.linalg.eigvalsh(S).min() > -1e-9 * np.abs(S).max()


def test_rebalancing_matrix_is_a_laplacian_with_nonnegative_weights():
    x = np.array([0.5, 0.3, 0.2])
    R = rebalancing_matrix(x)
    assert np.allclose(R.sum(axis=1), 0) and np.linalg.eigvalsh(R).min() > -1e-12
    assert (R - np.diag(np.diag(R)) <= 1e-15).all()


def test_baseline_cost_scale_inverts_the_stable_root():
    for phi in (0.2, 0.5, 0.9):
        beta_k = BETA_DAILY * 0.9
        cost = baseline_cost_scale(1.0, phi, beta_k)
        assert stable_root(np.array([beta_k]), np.array([1.0 / cost]))[0] == pytest.approx(phi)


def test_stable_root_is_in_unit_interval_and_matches_garleanu_pedersen():
    beta = np.array([0.99, 0.9996, 0.95])
    g = np.array([0.01, 3.0, 400.0])
    phi = stable_root(beta, g)
    assert ((phi > 0) & (phi < 1)).all()
    assert np.allclose(beta * phi**2 - (1 + beta + g) * phi + 1, 0, atol=1e-12)
    # Garleanu-Pedersen (2013) eq. (9): trading rate a / lambda equals 1 - phi
    for gamma, lam, rg in [(25 * 4e-4, 0.1, 0.0004), (5.0, 0.3, 0.05), (2.0, 10.0, 0.001)]:
        b = 1 - rg
        s = gamma * (1 - rg) / lam
        root = stable_root(np.array([b]), np.array([s]))[0]
        a = (-(gamma * (1 - rg) + lam * rg) + np.sqrt((gamma * (1 - rg) + lam * rg) ** 2 + 4 * gamma * lam * (1 - rg) ** 2)) / (2 * (1 - rg))
        assert 1 - root == pytest.approx(a / lam, rel=1e-10)


def test_covariance_estimate_is_positive_definite_and_checks_sample_size():
    rng = np.random.default_rng(0)
    returns = rng.normal(size=(300, 20))
    cov = estimate_covariance(returns, ridge=1e-8)
    assert np.linalg.eigvalsh(cov).min() > 0
    with pytest.raises(GameError):
        estimate_covariance(rng.normal(size=(5, 20)), ridge=1e-8)


def test_amihud_liquidity_is_inverse_to_illiquidity_and_winsorised():
    rng = np.random.default_rng(1)
    abs_ret = np.abs(rng.normal(size=(60, 50))) * 0.01
    dv = rng.uniform(1e6, 1e8, size=(60, 50))
    ell = amihud_liquidity(abs_ret, dv, scale=1.0, low=0.01, high=0.99)
    assert (ell > 0).all() and ell.shape == (50,)
    with pytest.raises(GameError):
        amihud_liquidity(abs_ret, np.zeros_like(dv), scale=1.0, low=0.01, high=0.99)


# --- static push -----------------------------------------------------------------------------


def test_agreement_without_rebalancing_gives_no_push(cfg):
    mk = random_market(seed=3, theta_zero=True)
    game = build_game(mk)
    knobs = neutral_knobs(game, cfg)
    belief = np.random.default_rng(0).normal(size=mk["N"]) * 0.01
    push = end_point(game, knobs, np.tile(belief, (mk["K"], 1)), belief)
    assert np.allclose(push.y, 0, atol=1e-14) and np.allclose(push.p, belief)


def test_conservation_without_sleeve_sum_of_liquidity_weighted_moves_is_zero(cfg):
    for seed in range(5):
        mk = random_market(seed=seed)
        game = build_game(mk)
        rng = np.random.default_rng(seed)
        push = end_point(game, neutral_knobs(game, cfg), rng.normal(size=(mk["K"], mk["N"])) * 0.01,
                         rng.normal(size=mk["N"]) * 0.01)
        assert abs(game.ell @ push.y) < 1e-12
        assert push.sleeve_flow == 0.0


def test_conservation_with_sleeve_equals_minus_the_sleeve_flow(cfg):
    for seed in range(5):
        mk = random_market(seed=seed, sleeve=True)
        game = build_game(mk)
        rng = np.random.default_rng(seed)
        push = end_point(game, neutral_knobs(game, cfg), rng.normal(size=(mk["K"], mk["N"])) * 0.01,
                         rng.normal(size=mk["N"]) * 0.01)
        assert game.ell @ push.y == pytest.approx(-push.sleeve_flow, abs=1e-12)
        assert abs(push.sleeve_flow) > 0


def test_sleeve_block_is_psd_laplacian_with_row_sums_equal_to_the_sleeve_weight(cfg):
    mk = random_market(seed=4, sleeve=True)
    game = build_game(mk)
    for fund in game.types:
        assert np.allclose(fund.S, fund.S.T)
        assert np.linalg.eigvalsh(fund.S).min() > -1e-9 * np.abs(fund.S).max()
        assert np.allclose(fund.S @ np.ones(mk["N"]), -fund.S_sleeve, atol=1e-9)
        assert np.allclose(fund.R @ np.ones(mk["N"]), -fund.R_sleeve, atol=1e-12)


def test_agreement_still_cancels_with_the_sleeve(cfg):
    mk = random_market(seed=5, sleeve=True, theta_zero=True)
    game = build_game(mk)
    belief = np.zeros(mk["N"])
    belief[0] = 0.02
    push = end_point(game, neutral_knobs(game, cfg), np.tile(belief, (mk["K"], 1)), belief)
    assert np.allclose(push.y, 0, atol=1e-13)


def test_pure_disagreement_pushes_coheld_neighbours_against_the_source(cfg):
    mk = random_market(seed=5, n=10, k=4, equicorr=True, theta_zero=True)
    game = build_game(mk)
    deltas = np.zeros((mk["K"], mk["N"]))
    deltas[:, 0] = 0.03
    delta0 = np.zeros(mk["N"])
    delta0[0] = 0.01
    push = end_point(game, neutral_knobs(game, cfg), deltas, delta0)
    assert push.y[0] > 0 and push.y[1:].max() < 0  # D12(vi)
    assert game.G.min() > 0  # Green's function of a positive-weight grounded Laplacian


def test_pure_rebalancing_pushes_neighbours_with_the_source(cfg):
    mk = random_market(seed=6, n=10, k=4, equicorr=True)
    mk["theta"] = np.ones(mk["K"])
    game = build_game(mk)
    belief = np.zeros(mk["N"])
    belief[0] = 0.02
    push = end_point(game, neutral_knobs(game, cfg), np.tile(belief, (mk["K"], 1)), belief)
    assert push.y[1:].min() > 0 or push.y[1:].max() < 0  # neighbours share one sign
    assert np.sign(push.y[1]) != np.sign(push.y[0]) or abs(push.y[0]) < abs(belief[0])


def test_types_pushes_sum_to_the_end_point_even_when_knobs_differ(cfg):
    mk = random_market(seed=7)
    game = build_game(mk)
    zeros = np.zeros(mk["K"])
    knobs = event_knobs(
        game, cfg.with_overrides({"game.kappa.fear": 0.5, "game.kappa.arousal": 0.5, "game.kappa.state": 0.5}),
        fear_minus_anger=np.array([0.8, -0.4, 0.2]), arousal=np.array([0.3, 0.9, -0.5]),
        state_z=np.array([1.0, -1.0, 0.5]), trait_b=np.array([0.5, -0.5, 0.2]),
    )
    rng = np.random.default_rng(7)
    push = end_point(game, knobs, rng.normal(size=(mk["K"], mk["N"])) * 0.01, rng.normal(size=mk["N"]) * 0.01)
    assert np.allclose(push.pi.sum(axis=0), push.y, atol=1e-14)
    assert not np.allclose(knobs.gamma, np.array([t.gamma_bar for t in game.types]))
    assert zeros.shape == (mk["K"],)


def test_neutral_knobs_equal_the_baseline(cfg):
    game = build_game(random_market(seed=8))
    knobs = neutral_knobs(game, cfg)
    assert np.allclose(knobs.gamma, [t.gamma_bar for t in game.types])
    assert np.allclose(knobs.f, 0)
    assert np.allclose(knobs.omega, [t.omega_bar for t in game.types])


def test_fear_raises_risk_aversion_and_arousal_lowers_cost(cfg):
    game = build_game(random_market(seed=9))
    on = cfg.with_overrides({"game.kappa.fear": 1.0, "game.kappa.arousal": 1.0})
    k = len(game.types)
    fear = event_knobs(game, on, fear_minus_anger=np.ones(k), arousal=np.zeros(k), state_z=np.zeros(k), trait_b=np.zeros(k))
    hot = event_knobs(game, on, fear_minus_anger=np.zeros(k), arousal=np.ones(k), state_z=np.zeros(k), trait_b=np.zeros(k))
    base = neutral_knobs(game, cfg)
    assert (fear.gamma > base.gamma).all() and (fear.omega < base.omega).all()
    assert (hot.cost < base.cost).all() and (hot.phi < base.phi).all()  # more urgent: trades faster


# --- conditional rule --------------------------------------------------------------------------


def test_conditional_rule_equals_direct_clearing_of_the_free_markets(cfg):
    mk = random_market(seed=10, n=12, k=4)
    game = build_game(mk)
    rng = np.random.default_rng(10)
    deltas, delta0 = rng.normal(size=(mk["K"], mk["N"])) * 0.01, rng.normal(size=mk["N"]) * 0.01
    push = end_point(game, neutral_knobs(game, cfg), deltas, delta0)
    pinned, observed = np.array([1, 4]), np.array([0.02, -0.01])
    cond = conditional_prices(game, push.c, delta0, pinned, observed)
    free = np.setdiff1d(np.arange(mk["N"]), pinned)
    A = game.A
    y_pinned = observed - delta0[pinned]
    direct = np.linalg.solve(A[np.ix_(free, free)], push.c[free] - A[np.ix_(free, pinned)] @ y_pinned)
    assert np.allclose(cond[free], delta0[free] + direct, atol=1e-12)
    assert np.allclose(cond[pinned], observed)
    G = game.G
    block = G[np.ix_(free, free)] - G[np.ix_(free, pinned)] @ np.linalg.solve(G[np.ix_(pinned, pinned)], G[np.ix_(pinned, free)])
    assert np.allclose(block, np.linalg.inv(A[np.ix_(free, free)]), atol=1e-10)  # block-inverse identity


def test_conditional_rule_reproduces_the_unconditional_one_when_the_move_is_the_prediction(cfg):
    mk = random_market(seed=11, n=9, k=3)
    game = build_game(mk)
    rng = np.random.default_rng(11)
    deltas, delta0 = rng.normal(size=(mk["K"], mk["N"])) * 0.01, rng.normal(size=mk["N"]) * 0.01
    push = end_point(game, neutral_knobs(game, cfg), deltas, delta0)
    pinned = np.array([2])
    cond = conditional_prices(game, push.c, delta0, pinned, push.p[pinned])
    assert np.allclose(cond, push.p, atol=1e-12)


def test_static_push_regressors_have_the_expected_shape_and_signs(cfg):
    mk = random_market(seed=12, n=10, k=4, equicorr=True)
    mk["theta"] = np.ones(mk["K"])
    game = build_game(mk)
    reb, sub = static_push_regressors(game, source=0, observed_move=0.02, type_belief=0.04)
    assert reb.shape == sub.shape == (mk["N"] - 1,)
    assert (reb >= 0).all() and (sub <= 0).all()  # substitution reads negative when delta_hat > s


# --- path -----------------------------------------------------------------------------------


def test_path_ends_at_the_static_end_point_and_starts_at_the_day0_share(cfg):
    mk = random_market(seed=13)
    game = build_game(mk)
    knobs = neutral_knobs(game, cfg)
    rng = np.random.default_rng(13)
    deltas, delta0 = rng.normal(size=(mk["K"], mk["N"])) * 0.01, rng.normal(size=mk["N"]) * 0.01
    push = end_point(game, knobs, deltas, delta0)
    path = price_path(push, knobs.phi, delta0, horizon=4000)
    assert np.allclose(path[-1], push.p, atol=1e-9)
    day0 = delta0 + ((1 - knobs.phi)[:, None] * push.pi).sum(axis=0)
    assert np.allclose(path[0], day0)


def test_expected_path_decays_toward_the_consensus_and_increments_difference_it(cfg):
    path = np.array([[0.04, -0.02], [0.05, -0.03], [0.05, -0.03]])
    consensus = np.zeros(2)
    kept = expected_path(path, consensus, varsigma=0.0)
    assert np.allclose(kept, path)
    faded = expected_path(path, consensus, varsigma=0.5)
    assert np.allclose(faded[0], path[0] * 0.5) and np.abs(faded[2]).max() < np.abs(path[2]).max()
    assert np.allclose(predicted_increments(kept), np.diff(path, axis=0))


def test_exact_dynamic_solution_converges_to_the_fast_path_as_impact_shrinks(cfg):
    """Spec 7.8 #1: the gap must shrink with impact (first-order approximation of D14b)."""
    mk = random_market(seed=14, n=6, k=3)
    rng = np.random.default_rng(14)
    deltas = rng.normal(size=(mk["K"], mk["N"])) * 0.01
    delta0 = rng.normal(size=mk["N"]) * 0.01
    Lam = [lam * mk["Sig"] for lam in mk["lam"]]
    gaps = []
    for scale in (10.0, 1.0, 1e-2, 1e-4, 1e-6):
        market = dict(mk, Lam=Lam)
        exact = exact_prices(market, deltas, delta0, horizon=300, psi_scale=scale)[:20]
        game = build_game(mk, psi_scale=scale)
        knobs = neutral_knobs(game, cfg)
        push = end_point(game, knobs, deltas, delta0)
        fast = price_path(push, knobs.phi, delta0, horizon=19)
        gaps.append(np.abs(exact - fast).max() / np.abs(exact - delta0).max())
    assert all(a > b for a, b in zip(gaps, gaps[1:]))  # the gap shrinks with impact
    assert gaps[-1] < 1e-2 and gaps[-1] < gaps[0] / 1e3


def test_exact_static_end_point_equals_the_fast_end_point(cfg):
    mk = random_market(seed=15, n=6, k=3)
    rng = np.random.default_rng(15)
    deltas = rng.normal(size=(mk["K"], mk["N"])) * 0.01
    delta0 = rng.normal(size=mk["N"]) * 0.01
    market = dict(mk, Lam=[lam * mk["Sig"] for lam in mk["lam"]])
    exact = exact_prices(market, deltas, delta0, horizon=1500, psi_scale=1.0)
    game = build_game(mk)
    push = end_point(game, neutral_knobs(game, cfg), deltas, delta0)
    assert np.abs(exact[-1] - push.p).max() < 1e-8


# --- anchor ----------------------------------------------------------------------------------


def test_anchor_reproduces_the_observed_day0_move(cfg):
    mk = random_market(seed=16, n=8, k=4)
    game = build_game(mk)
    knobs = neutral_knobs(game, cfg)
    rng = np.random.default_rng(16)
    deltas, delta0 = rng.normal(size=(mk["K"], mk["N"])) * 0.01, rng.normal(size=mk["N"]) * 0.01
    pinned = np.array([0, 3])
    model_day0 = price_path(end_point(game, knobs, deltas, delta0), knobs.phi, delta0, 1)[0][pinned]
    observed = model_day0 + np.array([0.003, -0.002])  # a modest surprise relative to the model
    path, anchor, _ = anchored_path(game, knobs, deltas, delta0, pinned, observed, cfg)
    assert not anchor.weak_id.any() and not anchor.clipped.any()
    assert np.allclose(path[0][pinned], observed, atol=1e-9)


def test_anchor_flags_weak_identification_and_clips(cfg):
    stuck = lambda tau: np.array([0.01, 0.02])  # price does not react to its own shift
    weak = anchor_outside_belief(stuck, np.array([0.5, 0.5]), cfg)
    assert weak.weak_id.all() and np.allclose(weak.tau, 0)
    steep = lambda tau: 1e-3 * tau  # needs a huge shift to hit the target
    clipped = anchor_outside_belief(steep, np.array([0.5]), cfg)
    assert clipped.clipped.all() and abs(clipped.tau[0]) == pytest.approx(0.10)


# --- volume and covariance ------------------------------------------------------------------


def test_volume_accounting_buys_equal_sells_and_net_is_liquidity_times_move(cfg):
    mk = random_market(seed=17, n=7, k=3)
    game = build_game(mk)
    knobs = neutral_knobs(game, cfg)
    rng = np.random.default_rng(17)
    deltas, delta0 = rng.normal(size=(mk["K"], mk["N"])) * 0.01, rng.normal(size=mk["N"]) * 0.01
    push = end_point(game, knobs, deltas, delta0)
    horizon = 30
    path = price_path(push, knobs.phi, delta0, horizon)
    u = fund_trades(push, knobs.phi, horizon)
    book = np.array([t.book for t in game.types])
    fund_net = np.einsum("k,ktn->tn", book, u)
    previous = np.vstack([delta0, path[:-1]])
    assert np.allclose(game.ell * (path - previous), fund_net, atol=1e-12)  # D17(ii)
    outside = -fund_net
    gross = np.einsum("k,ktn->tn", book, np.abs(u))
    buys = np.einsum("k,ktn->tn", book, np.clip(u, 0, None)) + np.clip(outside, 0, None)
    sells = np.einsum("k,ktn->tn", book, np.clip(-u, 0, None)) + np.clip(-outside, 0, None)
    assert np.allclose(buys, sells)  # D17(i)
    assert np.allclose(buys, 0.5 * (gross + np.abs(outside)))
    out = event_volume(game, push, knobs.phi, path, delta0, np.full(mk["N"], 1.0), window_days=6)
    assert out["volume_dial"].shape == (mk["N"],) and ((out["one_sided_share"] >= 0) & (out["one_sided_share"] <= 1)).all()


def test_covariance_shift_is_psd_and_zero_for_a_constant_increment():
    delta0 = np.zeros(3)
    path = np.cumsum(np.tile([0.01, -0.02, 0.0], (12, 1)), axis=0)
    shift = covariance_shift(path, delta0, horizon=10)
    assert np.allclose(shift, 0, atol=1e-18)
    rng = np.random.default_rng(0)
    wiggly = np.cumsum(rng.normal(size=(12, 3)) * 0.01, axis=0)
    assert np.linalg.eigvalsh(covariance_shift(wiggly, delta0, horizon=10)).min() > -1e-15


def test_strength_normalisation_sets_the_median_sensitivity_to_liquidity_ratio(cfg):
    from oryxis.mfg.precompute import FundInput, build_month_game, normalise_liquidity

    rng = np.random.default_rng(21)
    n = 12
    factors = rng.normal(size=(n + 1, 2)) * 0.01
    sigma = factors @ factors.T + np.diag(rng.uniform(1e-4, 4e-4, n + 1))
    funds = [FundInput(f"f{k}", ["mutual_pension", "bank_dealer", "hedge_fund"][k % 3], float(10 ** rng.uniform(9, 10)),
                       rng.dirichlet(np.ones(n)) * 0.8, 0.5) for k in range(4)]
    ell_raw = 10 ** rng.uniform(9, 11, n)
    for target in (1.0, 0.25):
        game = build_month_game(cfg.with_overrides({"game.strength_target": target}), [f"a{i}" for i in range(n)], sigma, ell_raw, funds)
        ratio = np.median(np.diag(game.M_bar) / game.ell)
        assert ratio == pytest.approx(target, rel=1e-9)
    raw = build_month_game(cfg.with_overrides({"game.strength_target": None}), [f"a{i}" for i in range(n)], sigma, ell_raw, funds)
    assert np.allclose(raw.ell, ell_raw)  # null keeps raw units (identity tests use this)
    assert normalise_liquidity(list(raw.types), ell_raw, 1.0).shape == ell_raw.shape
