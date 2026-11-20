"""Count of tuned or varied configurations that the new pieces add (spec 11.5): printed beside every headline and fed to the
deflated Sharpe. Both arms of the dial map count, and so does every merge rule and brake that is compared."""

from __future__ import annotations

from oryxis.common.config import Config


def dial_map_configurations(cfg: Config) -> int:
    """Ridge grid points, plus the tree grid points when the LightGBM arm is on (the same CV and budget apply to both)."""
    spec = cfg.get("reader.dials")
    count = len(spec["ridge_grid"]) if "ridge" in spec["arms"] else 0
    if "lgbm" in spec["arms"] and spec["lgbm"]["enabled"]:
        count += len(spec["lgbm"]["grid"])
    return count


def merge_configurations(cfg: Config) -> int:
    """Rules in the dev race x trust ridge grid points for the context-weighted rule, plus the brake variants compared."""
    merge, brake = cfg.get("framework.merge"), cfg.get("framework.brake")
    rules = len(merge["candidates"]) - (1 if "lcb" in merge["candidates"] else 0)
    lcb = len(merge["ridge_grid"]) if "lcb" in merge["candidates"] else 0
    return rules + lcb + len(brake["candidates"])
