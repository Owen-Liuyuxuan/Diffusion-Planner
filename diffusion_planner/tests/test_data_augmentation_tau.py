from __future__ import annotations

import numpy as np
import pytest
import torch
from diffusion_planner.utils.data_augmentation_tau import (
    KinematicLimits,
    TauOffsetBump,
    _offset_weight_derivs,
    _valid_future_merges,
)


def _make_tau_augmentor(**kwargs) -> TauOffsetBump:
    defaults = {
        "augment_prob": 1.0,
        "num_refine": 20,
        "device": "cpu",
        "ego_past_noise_std": 0.0,
        "use_smoothing_future_trajectory": False,
        "tau_min_s": 0.0,
        "tau_max_s": 0.0,
        "tau_lon_m": 0.4,
        "tau_lat_m": 0.5,
        "tau_dense_dt": 0.02,
    }
    defaults.update(kwargs)
    return TauOffsetBump(**defaults)


def _straight_scene(
    *,
    speed: float = 5.0,
    current_xy: tuple[float, float] = (0.0, 0.0),
    past_len: int = 31,
    future_len: int = 80,
    dt: float = 0.1,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    cx, cy = current_xy
    ego_past = torch.zeros(1, past_len, 4)
    ego_past[0, :, 0] = cx + torch.arange(-(past_len - 1), 1) * dt * speed
    ego_past[0, :, 1] = cy
    ego_past[0, :, 2] = 1.0
    ego_future = torch.zeros(1, future_len, 3)
    ego_future[0, :, 0] = cx + torch.arange(1, future_len + 1) * dt * speed
    ego_future[0, :, 1] = cy
    inputs = {
        "ego_current_state": torch.tensor([[cx, cy, 1.0, 0.0, speed, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "ego_agent_past": ego_past,
        "ego_shape": torch.tensor([[2.75, 5.0, 2.0]]),
        "goal_pose": torch.zeros(1, 4),
        "neighbor_agents_past": torch.zeros(1, 32, 31, 11),
        "lanes": torch.zeros(1, 140, 20, 33),
        "route_lanes": torch.zeros(1, 25, 20, 33),
        "polygons": torch.zeros(1, 10, 40, 2),
        "line_strings": torch.zeros(1, 10, 20, 4),
        "static_objects": torch.zeros(1, 5, 10),
    }
    return inputs, ego_future


def test_single_interpolant_ends_and_bias() -> None:
    t_left, t_b, t_merge = -2.0, -0.4, 2.0
    t = np.array([t_left - 0.1, t_left, t_b, 0.0, t_merge, t_merge + 0.5])
    w, dw, ddw, _ = _offset_weight_derivs(t, t_left, t_b, t_merge)
    assert w[t == t_left] == pytest.approx(0.0, abs=1e-6)
    assert w[t == t_merge] == pytest.approx(0.0, abs=1e-6)
    assert w[t == t_b] == pytest.approx(1.0, abs=1e-6)
    assert dw[t == t_left] == pytest.approx(0.0, abs=1e-6)
    assert dw[t == t_merge] == pytest.approx(0.0, abs=1e-6)
    assert ddw[t == t_left] == pytest.approx(0.0, abs=1e-6)
    assert ddw[t == t_merge] == pytest.approx(0.0, abs=1e-6)
    assert abs(float(dw[t == t_b][0])) > 1e-3


def test_weight_may_exceed_one_off_midpoint() -> None:
    t_left, t_b, t_merge = -2.0, -1.5, 2.0
    t = np.linspace(t_left, t_merge, 200)
    w, _, _, _ = _offset_weight_derivs(t, t_left, t_b, t_merge)
    assert float(np.max(w)) > 1.0 + 1e-3


def test_straight_line_searches_merge_and_rejoins_gt() -> None:
    torch.manual_seed(3)
    inputs, ego_future = _straight_scene(speed=6.0)
    orig_future = ego_future.clone()
    aug = _make_tau_augmentor(num_refine=20, tau_merge_slack_s=1.0)
    flag, aug_current, aug_past, aug_future = aug.augment_at_tau(inputs, ego_future.clone())
    assert flag.tolist() == [True]
    info = aug.last_bump_info
    assert info is not None
    t_merge = float(info["t_merge_s"])
    t_star = float(info["t_star_s"])
    t_max = float(info["t_search_max_s"])
    assert info["t_b_s"] == pytest.approx(0.0)
    assert info["t_left_s"] == pytest.approx(info["t_b_s"] - 2.0)
    assert info["history_merge_s"] == pytest.approx(info["t_b_s"] - 2.0)
    assert t_star >= 0.5
    assert t_star <= t_merge <= t_max + 1e-9
    assert t_max == pytest.approx(3.0)
    assert info["merge_pos_err_m"] < 1e-3
    merge_idx = int(round(t_merge / 0.1)) - 1
    assert torch.allclose(aug_future[0, merge_idx:, :2], orig_future[0, merge_idx:, :2], atol=1e-4)
    assert torch.isfinite(aug_past).all()
    assert torch.isfinite(aug_current).all()


def test_small_bias_can_merge_before_two_seconds() -> None:
    torch.manual_seed(0)
    inputs, ego_future = _straight_scene(speed=6.0)
    aug = _make_tau_augmentor(tau_lon_m=0.2, tau_lat_m=0.2, tau_merge_slack_s=0.0)
    flag, _, _, _ = aug.augment_at_tau(inputs, ego_future.clone())
    assert flag.tolist() == [True]
    assert aug.last_bump_info["t_star_s"] >= 0.5
    assert aug.last_bump_info["t_star_s"] < 2.0
    assert aug.last_bump_info["t_merge_s"] == pytest.approx(aug.last_bump_info["t_star_s"])


def test_peak_can_sit_in_history() -> None:
    torch.manual_seed(4)
    inputs, ego_future = _straight_scene(speed=6.0)
    aug = _make_tau_augmentor(tau_min_s=-1.0, tau_max_s=-0.3)
    flag, _, _, _ = aug.augment_at_tau(inputs, ego_future.clone())
    assert flag.tolist() == [True]
    info = aug.last_bump_info
    assert info["t_b_s"] < 0.0
    assert info["t_b_s"] >= -1.0
    t_hist = -3.0
    assert info["history_merge_s"] == pytest.approx(max(t_hist, float(info["t_b_s"]) - 2.0))
    assert info["t_merge_s"] >= 0.5
    assert abs(info["dw_at_0"]) > 1e-3


def test_bias_at_zero_offsets_current_pose() -> None:
    torch.manual_seed(5)
    inputs, ego_future = _straight_scene(speed=8.0)
    aug = _make_tau_augmentor(tau_lon_m=0.5, tau_lat_m=0.5, tau_min_s=0.0, tau_max_s=0.0)
    flag, aug_current, _, _ = aug.augment_at_tau(inputs, ego_future.clone())
    assert flag.tolist() == [True]
    assert torch.linalg.norm(aug_current[0, :2]) > 0.05
    assert abs(aug.last_bump_info["dw_at_0"]) > 1e-4


def test_centric_transform_recenters_current_pose() -> None:
    torch.manual_seed(7)
    inputs, ego_future = _straight_scene(speed=5.0)
    neighbors_future = torch.zeros(1, 32, 80, 4)
    aug = _make_tau_augmentor()
    out_inputs, out_future, _ = aug(inputs, ego_future.clone(), neighbors_future.clone())
    assert torch.linalg.norm(out_inputs["ego_current_state"][0, :2]) < 1e-4
    heading = torch.atan2(
        out_inputs["ego_current_state"][0, 3],
        out_inputs["ego_current_state"][0, 2],
    )
    assert abs(float(heading.item())) < 0.05


def test_dwell_history_does_not_flip_heading() -> None:
    """Repeated past poses must not get a π heading from cubic-spline velocity."""
    torch.manual_seed(0)
    inputs, ego_future = _straight_scene(speed=3.0)
    inputs["ego_agent_past"][0, :3] = inputs["ego_agent_past"][0, 3]
    orig_h = torch.atan2(inputs["ego_agent_past"][0, :, 3], inputs["ego_agent_past"][0, :, 2])
    aug = _make_tau_augmentor()
    flag, _, aug_past, _ = aug.augment_at_tau(inputs, ego_future.clone())
    assert flag.tolist() == [True]
    h = torch.atan2(aug_past[0, :, 3], aug_past[0, :, 2])
    dh = (h.diff() + np.pi) % (2 * np.pi) - np.pi
    assert float(dh.abs().max()) < 0.5
    assert torch.allclose(h[:3], orig_h[:3], atol=1e-3)


def _bump_window_ok(
    t: np.ndarray,
    base_vel: np.ndarray,
    base_acc: np.ndarray,
    base_jerk: np.ndarray,
    bias: np.ndarray,
    t_left: float,
    t_b: float,
    t_merge: float,
    limits: KinematicLimits,
) -> bool:
    valid = _valid_future_merges(
        t_left,
        np.array([t_merge], dtype=np.float64),
        t,
        base_vel,
        base_acc,
        base_jerk,
        bias,
        t_b,
        limits,
    )
    return bool(valid[0])


def test_validate_accepts_small_bump_on_straight_line() -> None:
    dt = 0.02
    t = np.arange(-3.0, 8.0 + 0.5 * dt, dt)
    base_vel = np.stack([np.full_like(t, 5.0), np.zeros_like(t)], axis=-1)
    base_acc = np.zeros_like(base_vel)
    base_jerk = np.zeros_like(base_vel)
    assert _bump_window_ok(
        t,
        base_vel,
        base_acc,
        base_jerk,
        np.array([0.5, 0.3]),
        t_left=-2.0,
        t_b=0.0,
        t_merge=2.0,
        limits=KinematicLimits(),
    )


def test_kappa_rate_rejects_short_violent_merge() -> None:
    dt = 0.02
    t = np.arange(-3.0, 8.0 + 0.5 * dt, dt)
    base_vel = np.stack([np.full_like(t, 8.0), np.zeros_like(t)], axis=-1)
    base_acc = np.zeros_like(base_vel)
    base_jerk = np.zeros_like(base_vel)
    bias = np.array([0.0, 0.4])
    limits = KinematicLimits()
    assert not _bump_window_ok(t, base_vel, base_acc, base_jerk, bias, -0.8, -0.2, 0.5, limits)
    assert _bump_window_ok(t, base_vel, base_acc, base_jerk, bias, -2.0, -0.2, 2.5, limits)


def test_torch_seed_reproducible_bias() -> None:
    inputs, ego_future = _straight_scene(speed=6.0)
    aug = _make_tau_augmentor()
    torch.manual_seed(11)
    _, _, _, fut_a = aug.augment_at_tau(inputs, ego_future.clone())
    info_a = dict(aug.last_bump_info)
    torch.manual_seed(11)
    _, _, _, fut_b = aug.augment_at_tau(inputs, ego_future.clone())
    info_b = dict(aug.last_bump_info)
    assert info_a["bias_lon_m"] == pytest.approx(info_b["bias_lon_m"])
    assert info_a["bias_lat_m"] == pytest.approx(info_b["bias_lat_m"])
    assert torch.allclose(fut_a, fut_b)


def test_rejected_augmentation_does_not_mutate_inputs() -> None:
    inputs, ego_future = _straight_scene()
    original_past = inputs["ego_agent_past"].clone()
    original_current = inputs["ego_current_state"].clone()
    aug = _make_tau_augmentor(tau_lon_m=0.0, tau_lat_m=0.0)

    def always_fail(self, **kwargs):
        return False

    aug._augment_single = always_fail.__get__(aug, TauOffsetBump)
    torch.manual_seed(0)
    flag, _, _, _ = aug.augment_at_tau(inputs, ego_future)
    assert flag.tolist() == [False]
    assert torch.equal(inputs["ego_agent_past"], original_past)
    assert torch.equal(inputs["ego_current_state"], original_current)


def test_random_seeds_acceptance_rate() -> None:
    accepted = 0
    for seed in range(30):
        torch.manual_seed(seed)
        inputs, ego_future = _straight_scene(
            current_xy=(float(seed % 3), float((seed % 5) - 2)),
            speed=4.0 + (seed % 4),
        )
        aug = _make_tau_augmentor()
        flag, _, _, _ = aug.augment_at_tau(inputs, ego_future.clone())
        if bool(flag[0].item()):
            accepted += 1
    assert accepted >= 20


def test_invalid_peak_range_raises() -> None:
    with pytest.raises(ValueError, match="history_lead_s"):
        _make_tau_augmentor(history_lead_s=0.0)
    with pytest.raises(ValueError, match="before min_future_merge"):
        _make_tau_augmentor(tau_min_s=0.0, tau_max_s=0.6, min_future_merge_s=0.5)


def test_legacy_constructor_positional_order() -> None:
    """Alias-compatible prefix: (prob, num_refine, device, noise, smoothing, ...)."""
    aug = TauOffsetBump(1.0, 20, "cpu", 0.1, False, -1.0, 0.0)
    assert aug._num_refine == 20
    assert aug._ego_past_noise_std == 0.1
    assert aug._use_smoothing_future_trajectory is False
    assert str(aug._device) == "cpu"


def test_zero_padded_history_rows_restored() -> None:
    torch.manual_seed(2)
    inputs, ego_future = _straight_scene(speed=6.0)
    inputs["ego_agent_past"][0, :5] = 0.0
    neighbors_future = torch.zeros(1, 32, 80, 4)
    aug = _make_tau_augmentor()
    out_inputs, _, _ = aug(inputs, ego_future.clone(), neighbors_future.clone())
    pad = out_inputs["ego_agent_past"][0, :5]
    assert torch.equal(pad, torch.zeros_like(pad))


def test_out_of_history_peak_rejects_without_crash() -> None:
    """Short history + tau_min before t_hist must not raise through the batch path."""
    torch.manual_seed(0)
    inputs, ego_future = _straight_scene(speed=6.0, past_len=6)  # t_hist = -0.5 s
    aug = _make_tau_augmentor(tau_min_s=-1.0, tau_max_s=-0.8)
    flag, _, _, _ = aug.augment_at_tau(inputs, ego_future.clone())
    assert flag.shape == (1,)
    # Empty feasible peak range after clamping => reject, not crash.
    assert flag.tolist() == [False]


def test_dense_grid_always_contains_zero() -> None:
    from diffusion_planner.utils.data_augmentation_tau import _dense_times_with_zero

    times = _dense_times_with_zero(-3.0, 8.0, 0.07)
    assert np.any(np.isclose(times, 0.0))


def test_heading_continuous_across_bump_boundary() -> None:
    torch.manual_seed(9)
    inputs, ego_future = _straight_scene(speed=6.0)
    aug = _make_tau_augmentor(tau_min_s=-1.0, tau_max_s=-0.3)
    flag, _, aug_past, aug_future = aug.augment_at_tau(inputs, ego_future.clone())
    assert flag.tolist() == [True]
    info = aug.last_bump_info
    assert info is not None
    past_h = torch.atan2(aug_past[0, :, 3], aug_past[0, :, 2]).numpy()
    fut_h = aug_future[0, :, 2].numpy()
    full_h = np.unwrap(np.concatenate([past_h, fut_h]))
    dh = np.diff(full_h)
    # No discrete jump larger than a few degrees per 0.1 s step.
    assert float(np.max(np.abs(dh))) < 0.2


def test_stopped_current_keeps_trajectory_heading() -> None:
    from diffusion_planner.utils.data_augmentation_tau import _current_state_from_kinematics

    state = _current_state_from_kinematics(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        omega0=0.0,
        wheel_base=2.75,
        dtype=torch.float32,
        device=torch.device("cpu"),
        heading=0.7,
        v_stop=0.2,
    )
    assert float(torch.atan2(state[3], state[2]).item()) == pytest.approx(0.7, abs=1e-5)


def test_smoothing_flag_forwarded() -> None:
    aug = _make_tau_augmentor(use_smoothing_future_trajectory=True)
    assert aug._use_smoothing_future_trajectory is True


def test_history_speed_noise_scales_past_about_current() -> None:
    inputs, _ = _straight_scene(speed=6.0, current_xy=(3.0, -1.0))
    orig_past = inputs["ego_agent_past"].clone()
    orig_current = inputs["ego_current_state"].clone()
    ego_current = orig_current.clone()
    ego_past = orig_past.clone()
    TauOffsetBump._scale_history_about_current(
        ego_current,
        ego_past,
        aug_flag=torch.tensor([True]),
        scale_by_batch=torch.tensor([1.2]),
    )
    center = orig_current[0, :2]
    expected_xy = center + (orig_past[0, :, :2] - center) * 1.2
    assert torch.allclose(ego_past[0, :, :2], expected_xy, atol=1e-5)
    assert torch.allclose(ego_current[0, 4:8], orig_current[0, 4:8] * 1.2, atol=1e-5)
