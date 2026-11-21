"""Optimiser (D21 and band), constraints, costs and the causal executor (spec 12.2 #11, #12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from finflock.portfolio.constraints import project, project_long_only, project_long_short
from finflock.portfolio.costs import (
    fixed_cost,
    lambda_base,
    lognormal_cost_multiplier,
    one_way_cost_fraction,
    sqrt_impact_cost,
)
from finflock.portfolio.execute import net_returns, run_backtest, scale_to_risk_target, turnover, vol_match
from finflock.portfolio.market import CausalityError, Market, MarketView
from finflock.portfolio.optimiser import (
    FirstStep,
    HorizonInputs,
    apply_band,
    constant_inputs,
    first_step,
    plan_path,
    stationary_value,
)

BETA = 0.9996


def random_inputs(seed, n=4, horizon=12):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(horizon, n)) * 2e-4
    sigma = np.empty((horizon, n, n))
    lam = np.empty((horizon, n, n))
    for t in range(horizon):
        f = rng.normal(size=(n, 2)) * 0.01
        sigma[t] = f @ f.T + np.diag(rng.uniform(1e-4, 4e-4, n))
        lam[t] = np.diag(rng.uniform(0.05, 0.2, n))
    return HorizonInputs(mu, sigma, lam)


def direct_qp(n_prev, inputs, gamma, beta):
    horizon, n = inputs.mu.shape
    H = np.zeros((n * horizon, n * horizon))
    f = np.zeros(n * horizon)
    for t in range(horizon):
        s = slice(t * n, (t + 1) * n)
        w = beta**t
        H[s, s] += w * (gamma * inputs.sigma[t] + inputs.lam[t])
        f[s] += w * inputs.mu[t]
        if t == 0:
            f[s] += inputs.lam[0] @ n_prev
        else:
            p = slice((t - 1) * n, t * n)
            H[p, p] += w * inputs.lam[t]
            H[s, p] -= w * inputs.lam[t]
            H[p, s] -= w * inputs.lam[t]
    return np.linalg.solve(H, f).reshape(horizon, n)


# --- optimiser -------------------------------------------------------------------------------


def test_backward_recursion_equals_the_direct_quadratic_programme():
    inputs = random_inputs(0)
    n_prev = np.random.default_rng(1).normal(size=4) * 0.05
    zeros = np.zeros((4, 4))
    path = plan_path(n_prev, inputs, 25.0, BETA, zeros, np.zeros(4))
    direct = direct_qp(n_prev, inputs, 25.0, BETA)
    assert np.allclose(path, direct, atol=1e-10)


def test_first_step_is_the_first_row_of_the_path():
    inputs = random_inputs(2)
    n_prev = np.zeros(4)
    step = first_step(n_prev, inputs, 10.0, BETA, np.zeros((4, 4)), np.zeros(4))
    assert np.allclose(step.n_free, plan_path(n_prev, inputs, 10.0, BETA, np.zeros((4, 4)), np.zeros(4))[0])


def test_zero_costs_give_the_myopic_mean_variance_solution():
    inputs = random_inputs(3)
    zero_lam = HorizonInputs(inputs.mu, inputs.sigma, np.zeros_like(inputs.lam))
    path = plan_path(np.zeros(4), zero_lam, 10.0, BETA, np.zeros((4, 4)), np.zeros(4))
    for t in range(inputs.mu.shape[0]):
        assert np.allclose(path[t], np.linalg.solve(10.0 * inputs.sigma[t], inputs.mu[t]), atol=1e-12)


def test_stationary_continuation_is_a_fixed_point():
    inputs = random_inputs(4)
    sigma, lam = inputs.sigma[0], inputs.lam[0]
    A = stationary_value(sigma, lam, 10.0, BETA, 1e-13, 100000)
    M = 10.0 * sigma + lam + BETA * A
    assert np.allclose(A, lam - lam @ np.linalg.solve(M, lam), atol=1e-11)


def test_stronger_costs_trade_less():
    inputs = random_inputs(5)
    n_prev = np.zeros(4)
    cheap = first_step(n_prev, inputs, 10.0, BETA, np.zeros((4, 4)), np.zeros(4)).n_free
    dear_inputs = HorizonInputs(inputs.mu, inputs.sigma, 20.0 * inputs.lam)
    dear = first_step(n_prev, dear_inputs, 10.0, BETA, np.zeros((4, 4)), np.zeros(4)).n_free
    assert np.linalg.norm(dear) < np.linalg.norm(cheap)


# --- band ------------------------------------------------------------------------------------


def scalar_step(mu, H, lam, n_prev):
    sigma = np.array([[H]])  # gamma = 1, A_next = 0
    inputs = HorizonInputs(np.array([[mu]]), sigma[None], np.array([[[lam]]]))
    return first_step(np.array([n_prev]), inputs, 1.0, BETA, np.zeros((1, 1)), np.zeros(1))


def test_band_matches_brute_force_in_the_scalar_one_period_case():
    rng = np.random.default_rng(6)
    worst = 0.0
    for _ in range(150):
        H, lam = rng.uniform(1e-3, 2e-2), rng.uniform(0.0, 0.2)
        kappa, mu, m = rng.uniform(1e-5, 1e-3), rng.normal() * 2e-4, rng.normal() * 0.05
        step = scalar_step(mu, H, lam, m)
        got = apply_band(np.array([m]), step, np.array([kappa]), band=1.0)[0]
        grid = np.linspace(m - 0.3, m + 0.3, 600001)
        objective = grid * mu - 0.5 * H * grid**2 - 0.5 * lam * (grid - m) ** 2 - kappa * np.abs(grid - m)
        worst = max(worst, abs(grid[np.argmax(objective)] - got))
    assert worst < 2e-6


def test_positions_inside_the_band_do_not_trade_and_band_zero_is_plain_d21():
    inputs = random_inputs(7, n=3)
    n_prev = np.array([0.02, -0.01, 0.0])
    step = first_step(n_prev, inputs, 10.0, BETA, np.zeros((3, 3)), np.zeros(3))
    wide = apply_band(n_prev, step, np.full(3, 1.0), band=1.0)  # enormous cost: everything inside
    assert np.allclose(wide, n_prev)
    assert np.allclose(apply_band(n_prev, step, np.full(3, 1e-4), band=0.0), step.n_free)
    narrow = apply_band(n_prev, step, np.full(3, 1e-9), band=1.0)
    assert np.allclose(narrow, step.n_free, atol=1e-6)


def test_inside_band_assets_hold_even_when_the_coupled_trade_would_move_them():
    """With correlated assets the D21 trade of an asset can exceed its band although its aim is inside
    the band; the rule is defined on the aim, so such an asset must still hold (R-8)."""
    rng = np.random.default_rng(13)
    decisive = 0
    for _ in range(300):
        inputs = random_inputs(int(rng.integers(1_000_000)), n=4, horizon=3)
        n_prev = rng.normal(size=4) * 0.02
        step = first_step(n_prev, inputs, 10.0, BETA, np.zeros((4, 4)), np.zeros(4))
        kappa = np.full(4, 3e-4)
        banded = apply_band(n_prev, step, kappa, band=1.0)
        diag_h = np.diag(step.H)
        inside = np.abs(step.aim - n_prev) <= kappa / diag_h
        coupled = np.abs(step.n_free - n_prev) - kappa / (diag_h + np.diag(step.lam0)) > 0
        decisive += int((inside & coupled).any())
        assert np.allclose(banded[inside], n_prev[inside])
    assert decisive > 0  # the case this test is about actually occurred


def test_band_never_trades_more_than_the_unconstrained_rule():
    inputs = random_inputs(8, n=5)
    n_prev = np.random.default_rng(8).normal(size=5) * 0.03
    step = first_step(n_prev, inputs, 10.0, BETA, np.zeros((5, 5)), np.zeros(5))
    banded = apply_band(n_prev, step, np.full(5, 5e-4), band=1.0)
    assert (np.abs(banded - n_prev) <= np.abs(step.n_free - n_prev) + 1e-15).all()
    assert (np.sign(banded - n_prev) * np.sign(step.n_free - n_prev) >= 0).all()


# --- constraints -----------------------------------------------------------------------------


def euclid(n, cap, max_gross):
    constraint = {"type": "ineq", "fun": lambda x: max_gross - x.sum()}
    res = minimize(lambda x: 0.5 * ((x - n) ** 2).sum(), np.zeros_like(n), bounds=[(0, cap)] * len(n),
                   constraints=[constraint], method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    return res.x


def test_long_only_projection_is_the_euclidean_projection(cfg):
    rng = np.random.default_rng(9)
    for _ in range(20):
        n = rng.normal(size=12) * 0.08
        got = project_long_only(n, 0.05, 0.4, 1e-14, 500)
        assert (got >= -1e-15).all() and (got <= 0.05 + 1e-15).all() and got.sum() <= 0.4 + 1e-9
        assert np.allclose(got, euclid(n, 0.05, 0.4), atol=1e-5)


def test_headline_projection_from_config_respects_cap_and_budget(cfg):
    n = np.random.default_rng(10).normal(size=100) * 0.2
    out = project(n, cfg, "headline")
    assert out.min() >= 0 and out.max() <= float(cfg.get("portfolio.constraints.headline.cap")) + 1e-12
    assert out.sum() <= float(cfg.get("portfolio.constraints.headline.max_gross")) + 1e-9
    assert np.allclose(project(out, cfg, "headline"), out)  # idempotent


def test_long_short_projection_is_dollar_neutral_with_gross_limit(cfg):
    n = np.random.default_rng(11).normal(size=40) * 0.2
    out = project(n, cfg, "long_short")
    spec = cfg.get("portfolio.constraints.long_short")
    assert abs(out.sum()) < 1e-9 and np.abs(out).max() <= spec["cap"] + 1e-12
    assert np.abs(out).sum() <= spec["max_gross"] + 1e-9
    assert np.allclose(project_long_short(np.zeros(5), 0.1, 2.0, 1e-14, 200), 0)


# --- costs -----------------------------------------------------------------------------------


def test_fixed_cost_is_half_the_round_trip_on_traded_notional(cfg):
    trade = np.array([0.10, -0.04])
    assert fixed_cost(trade, 10.0, 1.0, cfg) == pytest.approx(0.14 * 5e-4)
    assert fixed_cost(trade, 10.0, 4.0, cfg) == pytest.approx(4 * 0.14 * 5e-4)
    assert one_way_cost_fraction(10.0, cfg) == pytest.approx(5e-4)


def test_lambda_base_reproduces_the_impact_cost_at_the_typical_trade(cfg):
    sigma, adv, aum, alpha = np.array([0.02]), np.array([5e7]), 1e8, 0.5
    lam_w = lambda_base(sigma, adv, alpha, aum, cfg)
    q_bar = float(cfg.get("portfolio.impact.typical_trade_adv_share")) * adv
    true_cost = sigma * (q_bar / adv) ** alpha * q_bar / aum  # share of AUM at the typical trade
    quad = 0.5 * lam_w * (q_bar / aum) ** 2
    assert np.allclose(quad, true_cost)


def test_sqrt_impact_cost_grows_with_trade_size_and_scales_with_multiplier(cfg):
    sigma, adv = np.array([0.02, 0.02]), np.array([5e7, 5e7])
    small = sqrt_impact_cost(np.array([0.01, 0.0]), sigma, adv, 1e8, 1.0, cfg)
    large = sqrt_impact_cost(np.array([0.04, 0.0]), sigma, adv, 1e8, 1.0, cfg)
    assert large == pytest.approx(small * 8)  # q^(3/2)
    assert sqrt_impact_cost(np.array([0.01, 0.0]), sigma, adv, 1e8, 2.0, cfg) == pytest.approx(2 * small)


def test_lognormal_multiplier_is_one_without_variance_and_grows_with_it():
    assert lognormal_cost_multiplier(0, 0, 0, 0.5) == 1.0
    assert lognormal_cost_multiplier(0.04, 0.0, 0.0, 0.5) > 1.0
    assert lognormal_cost_multiplier(0.04, 0.04, 0.04, 0.5) == pytest.approx(np.exp(0.5 * (0.04 + 0.01 - 0.04)))


# --- executor and causality ----------------------------------------------------------------


def make_market(seed=0, n_days=60, names=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-02", periods=n_days)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, len(names))), axis=0)), index=index, columns=list(names))
    volume = pd.DataFrame(1e6, index=index, columns=list(names))
    oo = close.pct_change().shift(-1)  # return from the open of row t to the next open (toy: close-based)
    return Market(close, volume, oo.iloc[:-1].reindex(index))


def momentum_target(date, view: MarketView, held):
    """A toy causal strategy: long the best 1-day performer, 5% cap."""
    last = view.close.pct_change().iloc[-1].dropna()
    return pd.Series({last.idxmax(): 0.05}) if len(last) else pd.Series(dtype=float)


def test_view_never_exposes_rows_after_the_decision_date():
    market = make_market()
    date = market.sessions[20]
    view = market.asof(date)
    assert view.close.index.max() == date and view.volume.index.max() == date
    with pytest.raises(CausalityError):
        MarketView(date, market._close, market._volume)


def test_positions_up_to_a_date_do_not_change_when_later_prices_change():
    base = make_market(seed=1)
    dates = list(base.sessions[5:40])
    first = run_backtest(base, dates, momentum_target, delist_return=None)
    cut = 20
    mutated_close = base._close.copy()
    mutated_close.iloc[cut + 1 :] *= 1.7  # change everything after decision date index `cut`
    mutated_oo = base._oo.copy()
    mutated_oo.iloc[cut + 1 :] += 0.05
    other = Market(mutated_close, base._volume, mutated_oo)
    second = run_backtest(other, dates, momentum_target, delist_return=None)
    horizon = first.positions.index[first.positions.index <= base.sessions[cut]]
    pd.testing.assert_frame_equal(first.positions.loc[horizon], second.positions.loc[horizon])


def test_gross_return_is_position_times_fill_return_and_cash_earns_rf():
    market = make_market(seed=2)
    dates = list(market.sessions[3:10])
    fixed = lambda d, v, h: pd.Series({"A": 0.5, "B": 0.25})
    rf = pd.Series(0.0002, index=market.sessions)
    result = run_backtest(market, dates, fixed, delist_return=None, risk_free_daily=rf)
    f0 = result.fill_dates[0]
    r = market.fill_return(f0)
    assert result.gross.iloc[0] == pytest.approx(0.5 * r["A"] + 0.25 * r["B"] + 0.25 * 0.0002)
    assert result.cash_weight.iloc[0] == pytest.approx(0.25)


def test_buy_and_hold_target_trades_only_the_drift_and_first_day_is_a_full_build():
    market = make_market(seed=3)
    dates = list(market.sessions[3:12])
    fixed = lambda d, v, h: pd.Series({"A": 0.5, "B": 0.5})
    result = run_backtest(market, dates, fixed, delist_return=None)
    assert result.trades.iloc[0].abs().sum() == pytest.approx(1.0)
    later = result.trades.iloc[1:].abs().sum(axis=1)
    assert (later < 0.05).all() and (later > 0).all()  # only rebalancing of drift
    assert (turnover(result).iloc[0]) == pytest.approx(0.5)


def test_net_returns_apply_each_cost_model_to_the_same_trades(cfg):
    market = make_market(seed=4)
    dates = list(market.sessions[3:15])
    result = run_backtest(market, dates, momentum_target, delist_return=None)
    models = {
        "bp10": lambda trade, d: fixed_cost(trade.values, 10.0, 1.0, cfg),
        "bp20": lambda trade, d: fixed_cost(trade.values, 20.0, 1.0, cfg),
    }
    net = net_returns(result, models)
    assert np.allclose((result.gross - net["bp10"]) * 2, result.gross - net["bp20"])
    assert (net["bp20"] <= net["bp10"] + 1e-15).all()


def test_delisting_applies_the_default_return_or_zero_and_is_counted():
    market = make_market(seed=5, names=("A", "B"))
    gone = market.sessions[8]
    oo = market._oo.copy()
    oo.loc[gone:, "B"] = np.nan
    delisted = Market(market._close, market._volume, oo, pd.Series({"B": market.sessions[7]}))
    target = lambda d, v, h: pd.Series({"A": 0.3, "B": 0.3})
    with_rule = run_backtest(delisted, list(market.sessions[5:12]), target, delist_return=-0.30)
    without = run_backtest(delisted, list(market.sessions[5:12]), target, delist_return=None)
    assert with_rule.delist_hits.sum() > 0
    assert with_rule.gross.min() < without.gross.min()
    assert with_rule.missing_return_hits.sum() == 0


def test_panel_end_is_not_a_delisting():
    """Every name still trades on the last session, whose open-to-open return is NaN for all (no next open): that is a cut of the data, not an exit."""
    market = make_market(seed=7, names=("A", "B"))
    oo = market._oo.copy()
    oo.iloc[-1] = np.nan
    ends = Market(market._close, market._volume, oo, pd.Series({"A": market.sessions[-1], "B": market.sessions[-1]}))
    target = lambda d, v, h: pd.Series({"A": 0.5, "B": 0.5})
    dates = list(market.sessions[-6:])
    with_rule = run_backtest(ends, dates, target, delist_return=-0.30)
    without = run_backtest(ends, dates, target, delist_return=None)
    assert with_rule.delist_hits.sum() == 0 and without.delist_hits.sum() == 0
    assert with_rule.fill_dates.max() < market.sessions[-1]  # the last session is never a fill
    assert np.allclose(with_rule.gross, without.gross)


def test_risk_target_scales_weights_so_volatility_hits_the_target_and_costs_scale_with_trades():
    market = make_market(seed=21, n_days=120)
    rf = pd.Series(0.0001, index=market.sessions)
    target = lambda d, v, h: pd.Series({"A": 0.2, "B": 0.2, "C": 0.1})
    dates = list(market.sessions[10:100])
    base = run_backtest(market, dates, target, delist_return=None, risk_free_daily=rf)
    scaled = scale_to_risk_target(base, 0.10, 252, rf)
    assert scaled.gross.std() * np.sqrt(252) == pytest.approx(0.10, rel=1e-9)
    c = scaled.risk_scale
    assert np.allclose(scaled.trades.to_numpy(), c * base.trades.to_numpy())
    again = run_backtest(market, dates, lambda d, v, h: c * target(d, v, h), delist_return=None, risk_free_daily=rf)  # the same book with the weights scaled up front
    assert np.allclose(scaled.gross.to_numpy(), again.gross.to_numpy(), atol=1e-12)
    assert np.allclose(scaled.cash_weight.to_numpy(), again.cash_weight.to_numpy(), atol=1e-12)
    models = {"bp10": lambda trade, d: 0.001 * float(np.abs(trade).sum())}
    cost_base = (base.gross - net_returns(base, models)["bp10"]).mean()
    cost_scaled = (scaled.gross - net_returns(scaled, models)["bp10"]).mean()
    assert cost_scaled == pytest.approx(c * cost_base, rel=1e-9)


def test_vol_matching_hits_the_target_volatility():
    returns = pd.Series(np.random.default_rng(12).normal(0.0005, 0.02, 500))
    scaled = vol_match(returns, 0.10, 252)
    assert scaled.std() * np.sqrt(252) == pytest.approx(0.10, abs=1e-12)
    assert isinstance(FirstStep, type) and isinstance(constant_inputs, type(lambda: 0))


# --- Lead, Correct, Brake (spec 9.1) ----------------------------------------------------------------------------------


def _rows(n=600, seed=11, trust_feature=True):
    from finflock.fit.return_weights import MergeRows

    rng = np.random.default_rng(seed)
    road1 = rng.normal(0, 1.0, n)
    raw2 = rng.normal(0, 1.0, n)
    z = rng.normal(0, 1.0, (n, 2))
    true_trust = 1.0 + (0.8 * z[:, 0] if trust_feature else 0.0)
    realised = road1 + np.clip(true_trust, 0, 2) * 0.6 * raw2 + rng.normal(0, 0.5, n)
    dates = pd.Series(pd.bdate_range("2021-01-04", periods=n))
    return MergeRows(road1, raw2, rng.normal(0, 1.0, n) + road1, realised, z, dates)


def test_trust_is_one_without_a_fit_and_stays_inside_its_range():
    from finflock.portfolio.combine import lead_correct, trust_value

    z = np.array([[3.0, -3.0], [0.0, 0.0]])
    assert np.allclose(trust_value(z, np.zeros(3), (0.0, 2.0)), 1.0)  # theta = 0 is the plain sum
    assert np.allclose(lead_correct(np.array([1.0]), np.array([2.0]), 1.0), 3.0)
    wild = trust_value(z, np.array([0.0, 5.0, 5.0]), (0.0, 2.0))
    assert wild.min() >= 0.0 and wild.max() <= 2.0


def test_trust_fit_shrinks_toward_one_with_a_big_ridge_and_the_plain_sum_is_always_in_the_race(cfg):
    from finflock.common.timing import TradingCalendar
    from finflock.fit.return_weights import fit_bucket, fit_merge
    from finflock.portfolio.combine import choose_by_loss

    cal = TradingCalendar("XNYS", "2020-01-01", "2023-12-31")
    rows = _rows()
    strong = cfg.with_overrides({"framework.merge.ridge_grid": [1.0e9]})
    model, _, _ = fit_bucket(rows, ["f0", "f1"], cal, strong)
    assert np.allclose(model.theta, 0.0, atol=1e-6)  # all the ridge, no freedom: trust 1
    fit = fit_merge([rows], ["f0", "f1"], cal, cfg)
    assert fit.rule in ("plain_sum", "half_average", "lcb") and "plain_sum" in fit.cv_loss
    with pytest.raises(ValueError):
        choose_by_loss({"half_average": 1.0, "lcb": 0.5})  # a race without the plain sum is refused
    assert choose_by_loss({"plain_sum": 1.0, "lcb": 1.0}) == "plain_sum"  # a tie goes to the plain sum


def test_a_real_trust_feature_is_kept_and_a_useless_one_is_not(cfg):
    from finflock.common.timing import TradingCalendar
    from finflock.fit.return_weights import fit_bucket

    cal = TradingCalendar("XNYS", "2020-01-01", "2023-12-31")
    _, kept, _ = fit_bucket(_rows(), ["f0", "f1"], cal, cfg)
    assert "f0" in kept and "f1" not in kept  # f0 drives the planted trust, f1 is noise (Giacomini-White screen)
    _, kept_none, _ = fit_bucket(_rows(trust_feature=False), ["f0", "f1"], cal, cfg)
    assert "f0" not in kept_none


def test_too_few_rows_fall_back_to_the_plain_sum(cfg):
    from finflock.common.timing import TradingCalendar
    from finflock.fit.return_weights import MergeRows, fit_merge

    cal = TradingCalendar("XNYS", "2020-01-01", "2023-12-31")
    tiny = MergeRows(*(np.ones(5) for _ in range(4)), np.ones((5, 2)), pd.Series(pd.bdate_range("2021-01-04", periods=5)))
    fit = fit_merge([tiny], ["f0", "f1"], cal, cfg)
    assert fit.rule == "plain_sum" and fit.models[0].slope == 1.0 and np.allclose(fit.models[0].trust(np.ones((3, 0))), 1.0)


def test_correction_slope_is_cross_fitted_so_a_rows_own_label_never_sets_its_slope(cfg):
    from finflock.common.timing import TradingCalendar
    from finflock.fit.return_weights import cross_fitted_correction

    cal = TradingCalendar("XNYS", "2020-01-01", "2023-12-31")
    rows = _rows()
    oof, _ = cross_fitted_correction(rows, cal, cfg)
    changed = _rows()
    changed.realised[:50] += 100.0  # the first block's labels change
    oof_changed, _ = cross_fitted_correction(changed, cal, cfg)
    assert np.allclose(oof[:50], oof_changed[:50])  # their own out-of-fold values do not move


def test_disagreement_is_relative_to_signal_size_and_the_three_brakes_behave():
    from finflock.portfolio.combine import abstain_mask, cautious_min, disagreement, pooled_variance, soft_threshold

    f1, f2 = np.array([1.0, 10.0, 1.0, -1.0]), np.array([0.5, 9.5, 1.0, 1.0])
    gap = disagreement(f1, f2, 1e-8)
    assert gap[2] == pytest.approx(0.0) and gap[0] == pytest.approx(0.5) and gap[1] < 0.06  # big news with the same absolute gap is not penalised
    assert np.allclose(cautious_min(f1, f2), [0.5, 9.5, 1.0, 0.0])  # smaller of the two, zero where the signs differ
    var = pooled_variance(0.4, 0.2, f1, f2)
    assert np.allclose(var, 0.3 + 0.25 * (f1 - f2) ** 2) and var[2] == pytest.approx(0.3)  # within average + between spread
    shrunk = soft_threshold(np.array([1.0, -1.0, 0.1]), np.array([0.3, 0.3, 0.3]), 1.0)
    assert np.allclose(shrunk, [0.7, -0.7, 0.0])  # toward zero, never past it
    assert abstain_mask(gap, 0.4).tolist() == [True, False, False, True]


def test_the_merge_modules_use_no_labels():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "finflock"
    for name in ("portfolio/combine.py", "fit/return_weights.py", "framework/run.py", "reader/state_features.py", "data/market.py"):
        tree = ast.parse((root / name).read_text())
        modules = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not [m for m in modules if m.split(".")[-1] == "labels"], name
