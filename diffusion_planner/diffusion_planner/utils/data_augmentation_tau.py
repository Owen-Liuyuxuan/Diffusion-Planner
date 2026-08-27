"""Tau augmentation: one C² offset from history merge to a searched future merge.

History merge is ``t_b - history_lead_s`` (default 2 s before the peak), clipped
to the first recorded history time. Sample ``(b, t_b)``, search future ``T``,
interpolate ``p̃ = p + w(t) b`` on that window, recenter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from diffusion_planner.utils.data_augmentation_geom import TIME_INTERVAL
from diffusion_planner.utils.data_augmentation_scene import (
    centric_transform,
    check_aug_validity,
    get_transform_matrix_batch,
)

_EPS = 1e-3
_PSI_B_MIN = 1e-8


@dataclass(frozen=True)
class KinematicLimits:
    """Comfort / feasibility limits on the bump window."""

    v_min: float = 0.0
    v_max: float = 25.0
    v_stop: float = 0.2
    a_lon_min: float = -3.0
    a_lon_max: float = 2.0
    a_lat_max: float = 1.5
    omega_max: float = 0.8
    # Excess |ω̇| over the base spline (rad/s²). Absolute comfort is ~2 rad/s².
    omega_rate_max: float = 0.4
    kappa_max: float = 0.1
    # |dκ/dt| of the bump, 1/m/s. Bicycle: κ̇ ≈ δ̇ / L; 0.08 /s ≈ 13 deg/s at the road wheel for L=2.75 m.
    kappa_rate_max: float = 0.1
    j_max: float = 5.0
    eps: float = _EPS


def _psi_and_derivs(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``ψ(u)=u³(1-u)³``; ``ψ=ψ'=ψ''=0`` at ``u=0`` and ``u=1``."""
    u = np.asarray(u, dtype=np.float64)
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    u5 = u4 * u
    u6 = u3 * u3
    psi = u3 - 3.0 * u4 + 3.0 * u5 - u6
    d1 = 3.0 * u2 - 12.0 * u3 + 15.0 * u4 - 6.0 * u5
    d2 = 6.0 * u - 36.0 * u2 + 60.0 * u3 - 30.0 * u4
    d3 = 6.0 - 72.0 * u + 180.0 * u2 - 120.0 * u3
    return psi, d1, d2, d3


def _kinematics_from_va(
    vel: np.ndarray,
    acc: np.ndarray,
    jerk: np.ndarray,
    limits: KinematicLimits,
) -> dict[str, np.ndarray]:
    speed = np.linalg.norm(vel, axis=-1)
    cross = vel[..., 0] * acc[..., 1] - vel[..., 1] * acc[..., 0]
    dot = vel[..., 0] * acc[..., 0] + vel[..., 1] * acc[..., 1]
    denom_v = speed + limits.eps
    denom_v2 = speed**2 + limits.eps
    denom_v3 = speed**3 + limits.eps
    return {
        "speed": speed,
        "a_lon": dot / denom_v,
        "a_lat": cross / denom_v,
        "omega": cross / denom_v2,
        "kappa": cross / denom_v3,
        "j_norm": np.linalg.norm(jerk, axis=-1),
    }


def _offset_weight_derivs(
    t: np.ndarray,
    t_left: float,
    t_b: float,
    t_merge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single C² interpolant on ``[t_left, T]`` with ``w(t_b)=1``.

    Ends match GT (``w=ẇ=ẅ=0``). The path cuts through ``b``; ``ẇ(t_b)`` is
    not forced to 0. ``|w|`` may exceed 1 if ``t_b`` is not the midpoint.
    """
    t = np.asarray(t, dtype=np.float64)
    w = np.zeros_like(t, dtype=np.float64)
    dw = np.zeros_like(t, dtype=np.float64)
    ddw = np.zeros_like(t, dtype=np.float64)
    dddw = np.zeros_like(t, dtype=np.float64)
    duration = t_merge - t_left
    if duration <= 0.0 or not (t_left < t_b < t_merge):
        return w, dw, ddw, dddw
    u_b = (t_b - t_left) / duration
    psi_b, _, _, _ = _psi_and_derivs(np.asarray(u_b))
    psi_b = float(np.reshape(psi_b, ()))
    if abs(psi_b) < _PSI_B_MIN:
        return w, dw, ddw, dddw
    mask = (t >= t_left) & (t <= t_merge)
    if not np.any(mask):
        return w, dw, ddw, dddw
    u = np.clip((t[mask] - t_left) / duration, 0.0, 1.0)
    psi, d1, d2, d3 = _psi_and_derivs(u)
    scale = 1.0 / psi_b
    w[mask] = psi * scale
    dw[mask] = d1 * scale / duration
    ddw[mask] = d2 * scale / duration**2
    dddw[mask] = d3 * scale / duration**3
    return w, dw, ddw, dddw


def _fit_base_spline(
    knot_times: np.ndarray,
    knot_xy: np.ndarray,
    dense_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cs_x = CubicSpline(knot_times, knot_xy[:, 0], bc_type="natural")
    cs_y = CubicSpline(knot_times, knot_xy[:, 1], bc_type="natural")
    pos = np.stack([cs_x(dense_times), cs_y(dense_times)], axis=-1)
    vel = np.stack([cs_x(dense_times, 1), cs_y(dense_times, 1)], axis=-1)
    acc = np.stack([cs_x(dense_times, 2), cs_y(dense_times, 2)], axis=-1)
    jerk = np.stack([cs_x(dense_times, 3), cs_y(dense_times, 3)], axis=-1)
    return pos, vel, acc, jerk


def _diff_last_axis(x: np.ndarray, dt: float) -> np.ndarray:
    """Forward difference along the last axis, first sample copied from the first step."""
    d = np.diff(x, axis=-1) / dt
    return np.concatenate([d[..., :1], d], axis=-1)


def _ok_kinematics(
    vel: np.ndarray,
    acc: np.ndarray,
    jerk: np.ndarray,
    base_vel: np.ndarray,
    base_acc: np.ndarray,
    base_jerk: np.ndarray,
    limits: KinematicLimits,
    dt: float,
) -> np.ndarray:
    """Pointwise feasibility. Jerk and κ̇ are excess over the base spline; ω/κ skipped when stopped."""
    kin = _kinematics_from_va(vel, acc, jerk, limits)
    kin_base = _kinematics_from_va(base_vel, base_acc, base_jerk, limits)
    excess_j = np.linalg.norm(jerk - base_jerk, axis=-1)
    moving = kin["speed"] >= limits.v_stop
    ok = (
        (kin["speed"] >= limits.v_min)
        & (kin["speed"] <= limits.v_max)
        & (kin["a_lon"] >= limits.a_lon_min)
        & (kin["a_lon"] <= limits.a_lon_max)
        & (np.abs(kin["a_lat"]) <= limits.a_lat_max)
        & (excess_j <= limits.j_max)
    )
    ok = ok & (~moving | (np.abs(kin["omega"]) <= limits.omega_max))
    ok = ok & (~moving | (np.abs(kin["kappa"]) <= limits.kappa_max))
    moving_step = moving.copy()
    moving_step[..., 1:] = moving[..., 1:] & moving[..., :-1]
    kappa_rate = _diff_last_axis(kin["kappa"], dt)
    kappa_rate_base = _diff_last_axis(kin_base["kappa"], dt)
    omega_rate = _diff_last_axis(kin["omega"], dt)
    omega_rate_base = _diff_last_axis(kin_base["omega"], dt)
    ok = ok & (~moving_step | (np.abs(kappa_rate - kappa_rate_base) <= limits.kappa_rate_max))
    ok = ok & (~moving_step | (np.abs(omega_rate - omega_rate_base) <= limits.omega_rate_max))
    base_speed = np.linalg.norm(base_vel, axis=-1)
    forward = vel[..., 0] * base_vel[..., 0] + vel[..., 1] * base_vel[..., 1]
    stopped = base_speed < limits.v_stop
    return ok & ((forward >= -limits.eps) | stopped)


def _valid_future_merges(
    history_merge_s: float,
    merge_candidates: np.ndarray,
    dense_t: np.ndarray,
    base_vel: np.ndarray,
    base_acc: np.ndarray,
    base_jerk: np.ndarray,
    bias: np.ndarray,
    t_b: float,
    limits: KinematicLimits,
) -> np.ndarray:
    """Vectorized feasibility of every future merge ``T`` for a fixed history merge."""
    k_count = merge_candidates.shape[0]
    n = dense_t.shape[0]
    duration = merge_candidates.reshape(k_count, 1) - history_merge_s
    u_b = (t_b - history_merge_s) / duration.reshape(k_count)
    psi_b, _, _, _ = _psi_and_derivs(u_b)
    good_b = np.abs(psi_b) >= _PSI_B_MIN
    t_broad = dense_t.reshape(1, n)
    t_merge_broad = merge_candidates.reshape(k_count, 1)
    in_win = (t_broad >= history_merge_s) & (t_broad <= t_merge_broad) & (duration > 0.0)
    u = np.zeros((k_count, n), dtype=np.float64)
    np.divide(t_broad - history_merge_s, duration, out=u, where=in_win)
    u = np.clip(u, 0.0, 1.0)
    psi, d1, d2, d3 = _psi_and_derivs(u)
    scale = np.where(good_b, 1.0 / psi_b, 0.0).reshape(k_count, 1)
    dw = np.where(in_win, d1 * scale / duration, 0.0)
    ddw = np.where(in_win, d2 * scale / duration**2, 0.0)
    dddw = np.where(in_win, d3 * scale / duration**3, 0.0)
    vel = base_vel.reshape(1, n, 2) + dw[:, :, None] * bias
    acc = base_acc.reshape(1, n, 2) + ddw[:, :, None] * bias
    jerk = base_jerk.reshape(1, n, 2) + dddw[:, :, None] * bias
    ok = _ok_kinematics(
        vel,
        acc,
        jerk,
        np.broadcast_to(base_vel.reshape(1, n, 2), vel.shape),
        np.broadcast_to(base_acc.reshape(1, n, 2), acc.shape),
        np.broadcast_to(base_jerk.reshape(1, n, 2), jerk.shape),
        limits,
        float(dense_t[1] - dense_t[0]) if n > 1 else 0.02,
    )
    valid = np.all(np.where(in_win, ok, True), axis=1) & good_b
    return valid


def _pick_merge_with_slack(
    merge_candidates: np.ndarray,
    valid: np.ndarray,
    slack_s: float,
) -> float | None:
    """Earliest feasible T, or a valid knot in ``[T*, T* + slack]``."""
    if not np.any(valid):
        return None
    valid_idx = np.flatnonzero(valid)
    t_star = float(merge_candidates[valid_idx[0]])
    if np.random.random() < 0.5:
        return t_star
    cap = t_star + slack_s
    pool = valid_idx[merge_candidates[valid_idx] <= cap + 1e-9]
    if pool.size == 0:
        return t_star
    return float(merge_candidates[int(np.random.choice(pool))])


def _vel_at_times(dense_t: np.ndarray, dense_vel: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Linear sample of a dense (N, 2) velocity series at ``times``."""
    return np.stack(
        [
            np.interp(times, dense_t, dense_vel[:, 0]),
            np.interp(times, dense_t, dense_vel[:, 1]),
        ],
        axis=-1,
    )


def _headings_from_velocity(
    vel: np.ndarray,
    *,
    v_stop: float = 0.2,
    seed_heading: float | None = None,
) -> np.ndarray:
    """``atan2`` of velocity, unwrapped. Frozen when ``|v|`` is below ``v_stop``.

    A natural cubic on a dwell (repeated poses) can reverse the interpolant
    velocity; freezing avoids a π heading flip that is not in the recorded data.
    Large single-step heading jumps (>\u03c0/2) are also frozen: they are not
    reachable under the comfort ω limit on the 0.1 s I/O grid and almost always
    mean a cubic dwell reversal rather than a real turn.
    """
    speed = np.linalg.norm(vel, axis=-1)
    headings = np.arctan2(vel[:, 1], vel[:, 0])
    if seed_heading is not None:
        # Prefer the recorded start even when the first sample is moving; the
        # bump then rotates continuously from that seed via unwrap.
        headings[0] = seed_heading
    for i in range(1, headings.shape[0]):
        step = (headings[i] - headings[i - 1] + np.pi) % (2.0 * np.pi) - np.pi
        if speed[i] < v_stop or abs(step) > 0.5 * np.pi:
            headings[i] = headings[i - 1]
    return np.unwrap(headings)


def _current_state_from_kinematics(
    pos0: np.ndarray,
    vel0: np.ndarray,
    acc0: np.ndarray,
    omega0: float,
    wheel_base: float,
    dtype: torch.dtype,
    device: torch.device,
    *,
    heading: float | None = None,
    v_stop: float = 0.2,
) -> torch.Tensor:
    """Build ``ego_current_state`` from XY kinematics.

    When speed is below ``v_stop``, ``heading`` (if given) is kept so the current
    pose stays aligned with the rewritten history / future rather than flipping
    to an arbitrary ``atan2`` of a near-zero velocity.
    """
    speed = float(np.linalg.norm(vel0))
    if speed > 1e-6:
        heading = float(np.arctan2(vel0[1], vel0[0]))
    elif heading is None:
        heading = 0.0
    if speed >= v_stop:
        steering = float(
            np.clip(
                np.arctan(omega0 * wheel_base / abs(speed)),
                -2.0 / 3.0 * np.pi,
                2.0 / 3.0 * np.pi,
            )
        )
    else:
        steering = 0.0
        omega0 = 0.0
    return torch.tensor(
        [
            float(pos0[0]),
            float(pos0[1]),
            np.cos(heading),
            np.sin(heading),
            float(vel0[0]),
            float(vel0[1]),
            float(acc0[0]),
            float(acc0[1]),
            steering,
            float(omega0),
        ],
        dtype=dtype,
        device=device,
    )


def _ego_past_pad_mask(ego_past_4d: np.ndarray) -> np.ndarray:
    """True where a history row is all-zero 4D padding (centric_transform contract)."""
    return np.sum(ego_past_4d[..., :4] != 0.0, axis=-1) == 0


def _dense_times_with_zero(t_hist: float, t_end: float, dense_dt: float) -> np.ndarray:
    """Uniform dense grid that always includes t=0 exactly."""
    times = np.arange(t_hist, t_end + 0.5 * dense_dt, dense_dt, dtype=np.float64)
    if not np.any(np.isclose(times, 0.0, atol=1e-12)):
        times = np.sort(np.concatenate([times, np.asarray([0.0], dtype=np.float64)]))
    return times


class TauOffsetBump:
    """``p̃ = p + w(t) b``: one C² interpolant from ``t_b - lead`` to ``T``."""

    min_bias_norm = 1e-3
    bias_sample_attempts = 3
    bias_shrinks = 2
    min_speed_mps = 2.0

    def __init__(
        self,
        augment_prob: float,
        num_refine: int,
        device: torch.device | str,
        ego_past_noise_std: float = 0.0,
        use_smoothing_future_trajectory: bool = False,
        tau_min_s: float = -1.0,
        tau_max_s: float = 0.0,
        tau_lon_m: float = 0.4,
        tau_lat_m: float = 0.5,
        tau_dense_dt: float = 0.02,
        tau_merge_slack_s: float = 1.0,
        history_lead_s: float = 2.0,
        min_future_merge_s: float = 0.5,
        kinematic_limits: KinematicLimits | None = None,
    ) -> None:
        """Match legacy ``StatePerturbationAtTau`` positional prefix for callers.

        Positional order is ``(augment_prob, num_refine, device, ego_past_noise_std,
        use_smoothing_future_trajectory, ...)`` so existing train wiring and the
        ``StatePerturbationAtTau`` alias keep working.
        """
        if tau_min_s > tau_max_s:
            raise ValueError("tau_min_s must be <= tau_max_s.")
        if history_lead_s <= 0.0:
            raise ValueError("history_lead_s must be positive.")
        if tau_max_s >= min_future_merge_s:
            raise ValueError(
                f"tau_max_s={tau_max_s} must be before min_future_merge_s={min_future_merge_s}."
            )
        self._augment_prob = float(augment_prob)
        self._num_refine = int(num_refine)
        self._device = torch.device(device)
        self._ego_past_noise_std = float(ego_past_noise_std)
        self._use_smoothing_future_trajectory = bool(use_smoothing_future_trajectory)
        self._tau_min_s = float(tau_min_s)
        self._tau_max_s = float(tau_max_s)
        self._tau_lon_m = float(tau_lon_m)
        self._tau_lat_m = float(tau_lat_m)
        self._tau_dense_dt = float(tau_dense_dt)
        self._tau_merge_slack_s = float(tau_merge_slack_s)
        self._history_lead_s = float(history_lead_s)
        self._min_future_merge_s = float(min_future_merge_s)
        self._limits = kinematic_limits or KinematicLimits()
        self.time_interval = TIME_INTERVAL
        self.last_bump_info: dict | None = None
        self.last_tau_info: dict | None = None
        self._debug = os.environ.get("TAU_AUG_DEBUG", "").lower() in ("1", "true", "yes")
        self.last_debug_fail: str | None = None
        self.last_aug_flag: torch.Tensor | None = None
        self.last_pre_centric_current: torch.Tensor | None = None

    def _merge_search_max(self, t_end: float) -> float:
        """Upper bound: quintic-length window plus slack, clipped to the trajectory."""
        return min(t_end, float(self._num_refine) * self.time_interval + self._tau_merge_slack_s)

    def search_merge_points(
        self,
        dense_t: np.ndarray,
        base_vel: np.ndarray,
        base_acc: np.ndarray,
        base_jerk: np.ndarray,
        bias: np.ndarray,
        t_b: float,
        t_hist: float,
        t_max: float,
        dt: float,
    ) -> tuple[float, float, float] | None:
        """History merge at ``t_b - lead`` (clipped to ``t_hist``); search future ``T``."""
        if t_b <= t_hist:
            raise ValueError(f"t_b={t_b} must be after the first history time {t_hist}.")
        history_merge_s = max(t_hist, t_b - self._history_lead_s)
        if history_merge_s >= t_b:
            raise ValueError(f"history merge {history_merge_s} must be before t_b={t_b}.")
        merge_cands = np.arange(self._min_future_merge_s, t_max + 0.5 * dt, dt)
        merge_cands = merge_cands[merge_cands > t_b]
        if merge_cands.size == 0:
            return None
        valid = _valid_future_merges(
            history_merge_s,
            merge_cands,
            dense_t,
            base_vel,
            base_acc,
            base_jerk,
            bias,
            t_b,
            self._limits,
        )
        if not np.any(valid):
            return None
        t_star = float(merge_cands[np.flatnonzero(valid)[0]])
        t_merge = _pick_merge_with_slack(merge_cands, valid, self._tau_merge_slack_s)
        if t_merge is None:
            return None
        return history_merge_s, float(t_merge), t_star

    def _log_debug(self, msg: str) -> None:
        if self._debug:
            print(f"[TauOffsetBump] {msg}")

    def __call__(self, inputs, ego_future, neighbors_future):
        aug_flag, aug_current, aug_past, aug_future = self._augment_batch(inputs, ego_future)
        self.last_aug_flag = aug_flag
        self.last_pre_centric_current = aug_current.detach().clone()
        inputs["ego_current_state"][aug_flag] = aug_current[aug_flag]
        inputs["ego_agent_past"][aug_flag] = aug_past[aug_flag]
        ego_future[aug_flag] = aug_future[aug_flag]
        return centric_transform(
            inputs,
            ego_future,
            neighbors_future,
            use_smoothing_future_trajectory=self._use_smoothing_future_trajectory,
            transform_ego_past=True,
        )

    def augment_at_tau(self, inputs, ego_future):
        return self._augment_batch(inputs, ego_future)

    def _augment_batch(self, inputs, ego_future):
        ego_current = inputs["ego_current_state"].clone()
        ego_past = inputs["ego_agent_past"].clone()
        aug_future = ego_future.clone()
        device = ego_current.device
        dtype = ego_current.dtype
        B = ego_current.shape[0]
        past_len = ego_past.shape[1]
        self.last_bump_info = None
        self.last_tau_info = None
        self.last_debug_fail = None

        valid_speed = torch.abs(ego_current[:, 4]) >= self.min_speed_mps
        aug_flag = (torch.rand(B, device=device) < self._augment_prob) & valid_speed
        if self._debug:
            vx = float(ego_current[0, 4].item())
            self._log_debug(
                f"batch b=0 vx={vx:.3f} valid_speed={bool(valid_speed[0].item())} "
                f"prob_roll={bool(aug_flag[0].item())}"
            )
            if not bool(valid_speed[0].item()):
                self.last_debug_fail = f"speed_gate vx={vx:.3f} < {self.min_speed_mps} m/s"

        # Legacy history-speed scale about the current pose, before the bump.
        history_scale = torch.ones(B, device=device, dtype=dtype)
        n_aug = int(aug_flag.sum().item())
        if n_aug > 0 and self._ego_past_noise_std > 0.0:
            W = self._ego_past_noise_std
            sampled = torch.normal(mean=1.0, std=W, size=(n_aug,), device=device).to(dtype=dtype)
            history_scale[aug_flag] = torch.clamp(sampled, 1.0 - 2 * W, 1.0 + 2 * W)
            self._scale_history_about_current(
                ego_current=ego_current,
                ego_past=ego_past,
                aug_flag=aug_flag,
                scale_by_batch=history_scale,
            )

        for batch_index in torch.nonzero(aug_flag, as_tuple=False).flatten():
            b = int(batch_index.item())
            try:
                ok = self._augment_single(
                    ego_current=ego_current,
                    ego_past=ego_past,
                    ego_future=aug_future,
                    wheel_base=float(inputs["ego_shape"][b, 0].item()),
                    batch_index=b,
                    past_len=past_len,
                )
            except (RuntimeError, ValueError) as exc:
                ok = False
                self.last_debug_fail = f"exception: {exc}"
                self._log_debug(f"b={b} {self.last_debug_fail}")
            if not ok:
                aug_flag[b] = False

        collision = self._check_aug_validity(ego_current, inputs)
        if self._debug and bool(collision[0].item()):
            self.last_debug_fail = "collision_check"
            self._log_debug("b=0 collision at t=0")
        aug_flag = aug_flag & ~collision
        if self.last_bump_info is not None:
            self.last_bump_info["accepted"] = bool(
                aug_flag[self.last_bump_info["batch_index"]].item()
            )
            self.last_tau_info = dict(self.last_bump_info)
        return aug_flag, ego_current, ego_past, aug_future

    @staticmethod
    def _scale_history_about_current(
        ego_current: torch.Tensor,
        ego_past: torch.Tensor,
        aug_flag: torch.Tensor,
        scale_by_batch: torch.Tensor,
    ) -> None:
        """Dilate accepted histories about current pose; restore all-zero 4D pads."""
        if not bool(aug_flag.any().item()):
            return
        pad_mask = None
        if ego_past.shape[-1] >= 4:
            pad_mask = torch.zeros(
                ego_past.shape[0],
                ego_past.shape[1],
                dtype=torch.bool,
                device=ego_past.device,
            )
            pad_mask[aug_flag] = torch.sum(torch.ne(ego_past[aug_flag, :, :4], 0), dim=-1) == 0
        scale_xy = scale_by_batch[aug_flag].reshape(-1, 1, 1)
        center_xy = ego_current[aug_flag, :2].unsqueeze(1)
        past_xy = ego_past[aug_flag, :, :2]
        ego_past[aug_flag, :, :2] = center_xy + (past_xy - center_xy) * scale_xy
        if pad_mask is not None:
            ego_past[pad_mask] = 0
        scale_state = scale_by_batch[aug_flag].reshape(-1, 1)
        ego_current[aug_flag, 4:8] *= scale_state

    def _check_aug_validity(self, aug_ego_state: torch.Tensor, inputs: dict) -> torch.Tensor:
        return check_aug_validity(aug_ego_state, inputs)

    def get_transform_matrix_batch(self, cur_state: torch.Tensor) -> torch.Tensor:
        return get_transform_matrix_batch(cur_state)

    def _sample_bias_and_peak(self, t_hist: float) -> tuple[np.ndarray, float] | None:
        """Sample ``(b, t_b)`` with ``t_b`` strictly after the first real history time."""
        bias = np.array(
            [
                np.random.uniform(-self._tau_lon_m, self._tau_lon_m),
                np.random.uniform(-self._tau_lat_m, self._tau_lat_m),
            ],
            dtype=np.float64,
        )
        # Keep a small gap so history_merge = max(t_hist, t_b - lead) stays < t_b.
        lo = max(self._tau_min_s, t_hist + self._tau_dense_dt)
        hi = self._tau_max_s
        if lo > hi:
            return None
        t_b = float(np.random.uniform(lo, hi))
        return bias, t_b

    def _augment_single(
        self,
        ego_current: torch.Tensor,
        ego_past: torch.Tensor,
        ego_future: torch.Tensor,
        wheel_base: float,
        batch_index: int,
        past_len: int,
    ) -> bool:
        dt = self.time_interval
        dense_dt = self._tau_dense_dt
        np.random.seed(int(torch.randint(0, 2**31 - 1, ()).item()))

        past_4d = ego_past[batch_index, :, :4].detach().cpu().numpy().astype(np.float64)
        pad_mask = _ego_past_pad_mask(past_4d)
        past_xy = past_4d[:, :2].copy()
        future_xy = ego_future[batch_index, :, :2].cpu().numpy().astype(np.float64)
        current_xy = ego_current[batch_index, :2].cpu().numpy()
        past_xy[-1] = current_xy
        # Current pose is never padding; keep the last knot even if the row was zero.
        pad_mask = pad_mask.copy()
        pad_mask[-1] = False
        valid_past = ~pad_mask
        if not np.any(valid_past):
            self.last_debug_fail = "no_valid_history_knots"
            return False

        past_times_all = (np.arange(past_len) - (past_len - 1)) * dt
        future_times = (np.arange(1, ego_future.shape[1] + 1)) * dt
        knot_xy = np.concatenate([past_xy[valid_past], future_xy], axis=0)
        knot_times = np.concatenate([past_times_all[valid_past], future_times])
        t_hist = float(knot_times[0])
        t_end = float(knot_times[-1])
        t_max = self._merge_search_max(t_end)

        dense_times = _dense_times_with_zero(t_hist, t_end, dense_dt)
        base_pos, base_vel, base_acc, base_jerk = _fit_base_spline(knot_times, knot_xy, dense_times)

        for sample_attempt in range(self.bias_sample_attempts):
            sampled = self._sample_bias_and_peak(t_hist)
            if sampled is None:
                self.last_debug_fail = (
                    f"sample_attempt={sample_attempt} empty_peak_range "
                    f"t_hist={t_hist:.3f} tau=[{self._tau_min_s}, {self._tau_max_s}]"
                )
                continue
            bias, t_b = sampled
            if float(np.linalg.norm(bias)) < self.min_bias_norm:
                self.last_debug_fail = (
                    f"sample_attempt={sample_attempt} bias_norm_below_min "
                    f"norm={float(np.linalg.norm(bias)):.4g}"
                )
                continue
            found = None
            for shrink in range(self.bias_shrinks):
                scale = 0.5**shrink
                trial = bias * scale
                if float(np.linalg.norm(trial)) < self.min_bias_norm:
                    break
                found = self.search_merge_points(
                    dense_times,
                    base_vel,
                    base_acc,
                    base_jerk,
                    trial,
                    t_b,
                    t_hist,
                    t_max,
                    dt,
                )
                if found is not None:
                    bias = trial
                    break
            if found is None:
                self.last_debug_fail = (
                    f"sample_attempt={sample_attempt} no_valid_window "
                    f"bias_norm={float(np.linalg.norm(bias)):.3f} t_b={t_b:.3f} "
                    f"t_max={t_max:.3f}"
                )
                continue

            history_merge_s, t_merge, t_star = found
            io_times = np.concatenate([past_times_all, future_times])
            # Dense spline omitted pads; rebuild full-length knot XY for I/O rewrite.
            knot_xy_io = np.concatenate([past_xy, future_xy], axis=0)
            self._write_bump_io(
                ego_current=ego_current,
                ego_past=ego_past,
                ego_future=ego_future,
                batch_index=batch_index,
                past_len=past_len,
                wheel_base=wheel_base,
                knot_xy=knot_xy_io,
                io_times=io_times,
                dense_times=dense_times,
                base_pos=base_pos,
                base_vel=base_vel,
                base_acc=base_acc,
                bias=bias,
                history_merge_s=history_merge_s,
                t_b=t_b,
                t_merge=t_merge,
                t_star=t_star,
                t_max=t_max,
                past_pad_mask=pad_mask,
            )
            return True

        self.last_debug_fail = self.last_debug_fail or "exhausted_bias_peak_attempts"
        self._log_debug(f"b={batch_index} {self.last_debug_fail}")
        return False

    def _recorded_headings(
        self,
        ego_past: torch.Tensor,
        ego_future: torch.Tensor,
        batch_index: int,
    ) -> np.ndarray:
        past_h = np.arctan2(
            ego_past[batch_index, :, 3].detach().cpu().numpy(),
            ego_past[batch_index, :, 2].detach().cpu().numpy(),
        )
        fut_np = ego_future[batch_index].detach().cpu().numpy()
        if fut_np.shape[-1] >= 4:
            fut_h = np.arctan2(fut_np[:, 3], fut_np[:, 2])
        else:
            fut_h = fut_np[:, 2]
        return np.concatenate([past_h, fut_h])

    def _write_bump_io(
        self,
        *,
        ego_current: torch.Tensor,
        ego_past: torch.Tensor,
        ego_future: torch.Tensor,
        batch_index: int,
        past_len: int,
        wheel_base: float,
        knot_xy: np.ndarray,
        io_times: np.ndarray,
        dense_times: np.ndarray,
        base_pos: np.ndarray,
        base_vel: np.ndarray,
        base_acc: np.ndarray,
        bias: np.ndarray,
        history_merge_s: float,
        t_b: float,
        t_merge: float,
        t_star: float,
        t_max: float,
        past_pad_mask: np.ndarray | None = None,
    ) -> None:
        """Resample offset XY / heading onto 0.1 s I/O and rewrite current state."""
        w_io, _, _, _ = _offset_weight_derivs(io_times, history_merge_s, t_b, t_merge)
        aug_xy = knot_xy + w_io[:, None] * bias
        w_dense, dw_dense, ddw_dense, _ = _offset_weight_derivs(
            dense_times, history_merge_s, t_b, t_merge
        )
        aug_vel_dense = base_vel + dw_dense[:, None] * bias
        aug_acc_dense = base_acc + ddw_dense[:, None] * bias
        aug_pos_dense = base_pos + w_dense[:, None] * bias

        # Evaluate current-state kinematics at exact t=0 (grid always contains 0).
        zero_hits = np.flatnonzero(np.isclose(dense_times, 0.0, atol=1e-12))
        idx0 = int(zero_hits[0]) if zero_hits.size else int(np.argmin(np.abs(dense_times)))
        pos0 = aug_pos_dense[idx0]
        vel0 = aug_vel_dense[idx0]
        acc0 = aug_acc_dense[idx0]
        kin0 = _kinematics_from_va(
            vel0.reshape(1, 2),
            acc0.reshape(1, 2),
            np.zeros((1, 2), dtype=np.float64),
            self._limits,
        )
        omega0 = float(kin0["omega"][0])

        orig_heading = self._recorded_headings(ego_past, ego_future, batch_index)
        knot_vel = _vel_at_times(dense_times, aug_vel_dense, io_times)
        # Derivative-based headings everywhere so bump boundaries stay continuous;
        # freeze uses the earliest recorded heading as the dwell seed.
        headings_io = _headings_from_velocity(
            knot_vel,
            v_stop=self._limits.v_stop,
            seed_heading=float(orig_heading[0]),
        )

        dev = ego_past.device
        dtyp = ego_past.dtype
        ego_past[batch_index, :, 0] = torch.from_numpy(aug_xy[:past_len, 0]).to(
            device=dev, dtype=dtyp
        )
        ego_past[batch_index, :, 1] = torch.from_numpy(aug_xy[:past_len, 1]).to(
            device=dev, dtype=dtyp
        )
        ego_past[batch_index, :, 2] = torch.cos(
            torch.from_numpy(headings_io[:past_len]).to(device=dev, dtype=dtyp)
        )
        ego_past[batch_index, :, 3] = torch.sin(
            torch.from_numpy(headings_io[:past_len]).to(device=dev, dtype=dtyp)
        )
        if past_pad_mask is not None and np.any(past_pad_mask):
            ego_past[batch_index, past_pad_mask] = 0
        ego_future[batch_index, :, 0] = torch.from_numpy(aug_xy[past_len:, 0]).to(
            device=dev, dtype=dtyp
        )
        ego_future[batch_index, :, 1] = torch.from_numpy(aug_xy[past_len:, 1]).to(
            device=dev, dtype=dtyp
        )
        ego_future[batch_index, :, 2] = torch.from_numpy(headings_io[past_len:]).to(
            device=dev, dtype=dtyp
        )
        heading0 = float(headings_io[past_len - 1])
        ego_current[batch_index] = _current_state_from_kinematics(
            pos0,
            vel0,
            acc0,
            omega0,
            wheel_base,
            dtype=ego_current.dtype,
            device=ego_current.device,
            heading=heading0,
            v_stop=self._limits.v_stop,
        )

        idx_merge = int(np.argmin(np.abs(dense_times - t_merge)))
        merge_err = float(np.linalg.norm(aug_pos_dense[idx_merge] - base_pos[idx_merge]))
        w0, dw0, _, _ = _offset_weight_derivs(np.array([0.0]), history_merge_s, t_b, t_merge)
        self.last_bump_info = {
            "batch_index": batch_index,
            "accepted": True,
            "t_left_s": history_merge_s,
            "history_merge_s": history_merge_s,
            "t_b_s": t_b,
            "t_merge_s": t_merge,
            "t_star_s": t_star,
            "t_search_max_s": t_max,
            "w_at_0": float(w0[0]),
            "dw_at_0": float(dw0[0]),
            "bias_lon_m": float(bias[0]),
            "bias_lat_m": float(bias[1]),
            "bias_norm_m": float(np.linalg.norm(bias)),
            "merge_pos_err_m": merge_err,
        }
        self.last_tau_info = dict(self.last_bump_info)
        self.last_debug_fail = None
        self._log_debug(
            f"b={batch_index} success bias_norm={float(np.linalg.norm(bias)):.3f} "
            f"history_merge={history_merge_s:.3f} t_b={t_b:.3f} "
            f"t_star={t_star:.3f} t_merge={t_merge:.3f}"
        )
