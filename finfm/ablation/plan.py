"""The ablation plan: switches, groups, ladder rungs, draws and noise-floor seeds read from ``config/ablation.yaml`` and expanded into runs.

A run key is one of ``A0`` (Full), ``<switch or group>``, ``<placebo switch>:<draw>``, ``<switch>@fixed:<draw>`` (forecast level only, fitted model kept),
``noise:<k>`` (Full refitted with another fit seed), ``persona:<fraction>``, ``L3`` / ``L4`` (ladder rungs). Nothing here reads data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from finfm.common.config import Config

FIXED_SUFFIX = "@fixed"
DRAW_SEPARATOR = ":"
PERCENT = 100
EFFECT_KEYS = ("readings", "links", "ripple", "answers", "consensus", "game", "drop_type", "merge_rule", "brake", "variant", "config", "check")


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    key: str
    kind: str  # full | switch | group | noise | persona | ladder
    name: str
    tier: str  # A | B
    mode: str  # fixed | refit
    refit: tuple[str, ...]
    effects: dict = field(default_factory=dict, hash=False)
    draw: int | None = None
    arm: str | None = None
    fraction: float | None = None
    placebo: bool = False
    status: str = "build"
    reason: str | None = None
    members: tuple[str, ...] = ()

    @property
    def identity(self) -> tuple:
        """The tuple every random draw of this run derives its seed from."""
        return ("ablation", self.name, self.draw if self.draw is not None else 0)


def _effects(entry: dict) -> dict:
    return {k: entry[k] for k in EFFECT_KEYS if k in entry}


def merge_effects(parts: list[dict]) -> dict:
    """The effects of the members of a group, applied together; two members setting the same effect differently is an error."""
    out: dict = {}
    for part in parts:
        for key, value in part.items():
            if key == "config":
                clash = set(out.get(key, {})) & set(value)
                if clash:
                    raise PlanError(f"two members of a group set the same config key: {sorted(clash)}")
                out[key] = {**out.get(key, {}), **value}
            elif key in out and out[key] != value:
                raise PlanError(f"two members of a group set {key} differently: {out[key]!r} and {value!r}")
            else:
                out[key] = value
    return out


class Plan:
    def __init__(self, cfg: Config, bucket: str, llm: str) -> None:
        self.cfg, self.bucket, self.llm = cfg, bucket, llm
        self.switches: dict = cfg.get("ablation.switches")
        self.groups: dict = cfg.get("ablation.groups")
        self.ladder: dict = cfg.get("ablation.ladder")

    def run_id(self, name: str, draw: int | None = None, suffix: str = "") -> str:
        return f"abl-{self.bucket}-{self.llm}-{name}{suffix}" + (f"-k{draw}" if draw is not None else "")

    def spec(self, key: str) -> RunSpec:
        """The run behind a key of config ``ablation.notebooks``; unknown keys raise."""
        if key == str(self.cfg.get("ablation.full.id")):
            return RunSpec(self.run_id("full"), key, "full", "full", "B", "fixed", ())
        base, _, rest = key.partition(DRAW_SEPARATOR)
        if base == str(self.cfg.get("ablation.noise_floor.seed_label")):
            return RunSpec(self.run_id("noise", int(rest)), key, "noise", "noise", "B", "refit", ("road1", "merge"), draw=int(rest))
        if base == "persona":
            return RunSpec(self.run_id("persona", suffix=f"-{int(round(float(rest) * PERCENT))}"), key, "persona", "persona_count", "A", "refit", ("merge",),
                           fraction=float(rest))
        if key in self.ladder:
            entry = self.ladder[key]
            return RunSpec(self.run_id(entry["name"]), key, "ladder", entry["name"], "B", entry.get("mode", "refit"), tuple(entry.get("refit", ["road1", "merge"])),
                           effects={**_effects(entry), "variant": entry["variant"]}, arm=entry.get("arm"))
        fixed = base.endswith(FIXED_SUFFIX)
        switch = base[: -len(FIXED_SUFFIX)] if fixed else base
        draw = int(rest) if rest else None
        if switch in self.groups:
            group = self.groups[switch]
            members = [self.switches[m] for m in group["members"]]
            refit = tuple(sorted({r for m in members for r in m.get("refit", [])}))
            mode = "refit" if any(m.get("mode") == "refit" for m in members) else "fixed"
            return RunSpec(self.run_id(group["name"]), key, "group", group["name"], "B", mode, refit, merge_effects([_effects(m) for m in members]),
                           members=tuple(group["members"]))
        if switch not in self.switches:
            raise PlanError(f"unknown run key {key!r}")
        entry = self.switches[switch]
        placebo = bool(entry.get("placebo"))
        if placebo and draw is None:
            raise PlanError(f"{key}: a placebo switch needs a draw, e.g. {key}:1")
        if fixed:
            return RunSpec(self.run_id(entry["name"], draw, "-fixed"), key, "switch", entry["name"], "A", "fixed", (), _effects(entry), draw, placebo=placebo)
        return RunSpec(self.run_id(entry["name"], draw), key, "switch", entry["name"], entry.get("tier", "B"), entry.get("mode", "fixed"),
                       tuple(entry.get("refit", [])), {**_effects(entry), **({"config": dict(entry["config"])} if "config" in entry else {})}, draw,
                       placebo=placebo, status=entry.get("status", "build"), reason=entry.get("reason"))

    def tier_a_keys(self) -> list[str]:
        """Forecast-level runs with the fitted model kept (spec section 7): K draws of each placebo switch, one run of every other listed switch."""
        draws = int(self.cfg.get("ablation.draws.tier_a"))
        keys = []
        for switch in self.cfg.get("ablation.tier_a_draws.switches"):
            if self.switches[switch].get("placebo"):
                keys += [f"{switch}{FIXED_SUFFIX}{DRAW_SEPARATOR}{k}" for k in range(1, draws + 1)]
            else:
                keys.append(f"{switch}{FIXED_SUFFIX}")
        return keys

    def notebook_runs(self) -> dict[str, list[RunSpec]]:
        """The runs of each notebook; a notebook with ``tier_a_draws: true`` runs the forecast-level draws first (seconds each), then its own list."""
        out = {}
        for name, entry in self.cfg.get("ablation.notebooks").items():
            keys = (self.tier_a_keys() if entry.get("tier_a_draws") else []) + list(entry["runs"])
            out[name] = [self.spec(k) for k in keys]
        return out

    def all_specs(self) -> list[RunSpec]:
        return [spec for specs in self.notebook_runs().values() for spec in specs]

    def not_computed(self) -> list[RunSpec]:
        """Switches of the spec that are listed but cannot be run, with the reason (the report prints them as 'not computed')."""
        return [self.spec_not_computed(k) for k, v in self.switches.items() if v.get("status") == "not_computed"]

    def spec_not_computed(self, key: str) -> RunSpec:
        entry = self.switches[key]
        return RunSpec(self.run_id(entry["name"]), key, "switch", entry["name"], entry.get("tier", "B"), "fixed", (), status="not_computed", reason=entry["reason"])
