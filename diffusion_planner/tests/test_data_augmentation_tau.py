from __future__ import annotations

import types

import pytest
import torch
from diffusion_planner.utils.data_augmentation import (
    StatePerturbationAtTau,
    polyline_tangential_va,
    tangential_va,
)


def _make_tau_augmentor(*, ego_past_noise_std: float = 0.1) -> StatePerturbationAtTau:
    return StatePerturbationAtTau(
        augment_prob=1.0,
        num_refine=20,
        device="cpu",
        ego_past_noise_std=ego_past_noise_std,
        use_smoothing_future_trajectory=False,
    )


def test_scale_history_about_nonzero_current_preserves_join() -> None:
    ego_current = torch.tensor(
        [
            [2.0, -1.0, 1.0, 0.0, 5.0, 0.5, -2.0, 1.0, 0.0, 0.0],
            [4.0, 3.0, 1.0, 0.0, 6.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    ego_past = torch.tensor(
        [
            [[0.0, -1.0, 1.0, 0.0], [1.0, -1.0, 1.0, 0.0], [2.0, -1.0, 1.0, 0.0]],
            [[2.0, 3.0, 1.0, 0.0], [3.0, 3.0, 1.0, 0.0], [4.0, 3.0, 1.0, 0.0]],
        ]
    )
    original_second_past = ego_past[1].clone()
    original_second_state = ego_current[1].clone()

    StatePerturbationAtTau._scale_history_about_current(
        ego_current=ego_current,
        ego_past=ego_past,
        aug_flag=torch.tensor([True, False]),
        scale_by_batch=torch.tensor([1.2, 0.8]),
    )

    expected_first_xy = torch.tensor([[-0.4, -1.0], [0.8, -1.0], [2.0, -1.0]])
    assert torch.allclose(ego_past[0, :, :2], expected_first_xy)
    assert torch.allclose(ego_past[0, -1, :2], ego_current[0, :2])
    assert torch.allclose(ego_current[0, 4:8], torch.tensor([6.0, 0.6, -2.4, 1.2]), atol=1e-6)
    assert torch.equal(ego_past[1], original_second_past)
    assert torch.equal(ego_current[1], original_second_state)


def test_tau_history_scale_is_applied_before_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    aug = _make_tau_augmentor(ego_past_noise_std=0.1)
    captured: dict[str, torch.Tensor | float] = {}

    def fixed_normal(*, mean, std, size, device):
        assert mean == 1.0
        assert std == pytest.approx(0.1)
        return torch.full(size, 1.2, device=device)

    def capture_bridge(self, **kwargs):
        b = kwargs["batch_index"]
        captured["past"] = kwargs["ego_past"][b].clone()
        captured["current"] = kwargs["ego_current"][b].clone()
        captured["history_scale"] = kwargs["history_scale"]
        return True

    monkeypatch.setattr(torch, "normal", fixed_normal)
    monkeypatch.setattr(
        aug,
        "_augment_single_at_tau",
        types.MethodType(capture_bridge, aug),
    )

    inputs = {
        "ego_current_state": torch.tensor([[2.0, -1.0, 1.0, 0.0, 5.0, 0.0, -2.0, 0.0, 0.0, 0.0]]),
        "ego_agent_past": torch.tensor(
            [[[0.0, -1.0, 1.0, 0.0], [1.0, -1.0, 1.0, 0.0], [2.0, -1.0, 1.0, 0.0]]]
        ),
        "ego_shape": torch.tensor([[2.75, 5.0, 2.0]]),
    }
    ego_future = torch.zeros(1, 8, 3)

    aug_flag, _, _, _ = aug.augment_at_tau(inputs, ego_future)

    assert aug_flag.tolist() == [True]
    assert captured["history_scale"] == pytest.approx(1.2)
    assert torch.allclose(
        captured["past"][:, :2],
        torch.tensor([[-0.4, -1.0], [0.8, -1.0], [2.0, -1.0]]),
    )
    assert torch.allclose(captured["current"][4:8], torch.tensor([6.0, 0.0, -2.4, 0.0]), atol=1e-6)


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
        "ego_current_state": torch.tensor(
            [[cx, cy, 1.0, 0.0, speed, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "ego_agent_past": ego_past,
        "ego_shape": torch.tensor([[2.75, 5.0, 2.0]]),
    }
    return inputs, ego_future


def _circle_scene(
    *,
    speed: float = 5.0,
    radius: float = 25.0,
    past_len: int = 31,
    future_len: int = 80,
    dt: float = 0.1,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Constant-speed left turn. t=0 is at `origin` with heading +x."""
    omega = speed / radius
    times_past = torch.arange(-(past_len - 1), 1) * dt
    times_future = torch.arange(1, future_len + 1) * dt
    ox, oy = origin

    def pose(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        heading = omega * t
        x = ox + radius * torch.sin(heading)
        y = oy + radius * (1.0 - torch.cos(heading))
        return x, y, heading

    px, py, ph = pose(times_past)
    fx, fy, fh = pose(times_future)
    ego_past = torch.stack([px, py, torch.cos(ph), torch.sin(ph)], dim=-1).unsqueeze(0)
    ego_future = torch.stack([fx, fy, fh], dim=-1).unsqueeze(0)
    vx = speed
    vy = 0.0
    ax = 0.0
    ay = speed * omega  # centripetal in +y at t=0
    inputs = {
        "ego_current_state": torch.tensor(
            [[ox, oy, 1.0, 0.0, vx, vy, ax, ay, 0.0, omega]]
        ),
        "ego_agent_past": ego_past,
        "ego_shape": torch.tensor([[2.75, 5.0, 2.0]]),
    }
    return inputs, ego_future


def _join_error_m(past: torch.Tensor, current: torch.Tensor) -> float:
    return float(torch.linalg.norm(past[0, -1, :2] - current[0, :2]).item())


def _max_step_m(past: torch.Tensor, current: torch.Tensor, future: torch.Tensor) -> float:
    xy = torch.cat([past[0, :, :2], current[0, :2].unsqueeze(0), future[0, :, :2]], dim=0)
    # current duplicates past[-1]; drop the duplicate for step stats
    xy = torch.cat([past[0, :, :2], future[0, :, :2]], dim=0)
    return float(torch.linalg.norm(xy[1:] - xy[:-1], dim=-1).max().item())


def _speed_jump_at_t0(
    past: torch.Tensor, current: torch.Tensor, future: torch.Tensor, dt: float = 0.1
) -> dict[str, float]:
    v_past = float(torch.linalg.norm(past[0, -1, :2] - past[0, -2, :2]).item() / dt)
    v_fut = float(torch.linalg.norm(future[0, 0, :2] - current[0, :2]).item() / dt)
    v_state = float(torch.linalg.norm(current[0, 4:6]).item())
    return {
        "v_past": v_past,
        "v_future": v_fut,
        "v_state": v_state,
        "past_vs_future": abs(v_past - v_fut),
        "state_vs_future": abs(v_state - v_fut),
    }


def test_scaled_real_tau_bridge_preserves_current_join() -> None:
    torch.manual_seed(7)
    dt = 0.1
    speed = 5.0
    past_len = 31
    future_len = 80

    ego_past = torch.zeros(1, past_len, 4)
    ego_past[0, :, 0] = torch.arange(-(past_len - 1), 1) * dt * speed
    ego_past[0, :, 2] = 1.0
    ego_future = torch.zeros(1, future_len, 3)
    ego_future[0, :, 0] = torch.arange(1, future_len + 1) * dt * speed
    inputs = {
        "ego_current_state": torch.tensor([[0.0, 0.0, 1.0, 0.0, speed, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "ego_agent_past": ego_past,
        "ego_shape": torch.tensor([[2.75, 5.0, 2.0]]),
    }
    aug = StatePerturbationAtTau(
        augment_prob=1.0,
        num_refine=20,
        device="cpu",
        ego_past_noise_std=0.1,
        use_smoothing_future_trajectory=False,
        tau_min_s=-0.5,
        tau_max_s=-0.5,
    )

    aug_flag, aug_current, aug_past, aug_future = aug.augment_at_tau(inputs, ego_future)

    assert aug_flag.tolist() == [True]
    assert aug.last_tau_info is not None
    assert 0.8 <= aug.last_tau_info["history_scale"] <= 1.2
    assert torch.allclose(aug_past[0, -1, :2], aug_current[0, :2], atol=1e-6)
    assert torch.isfinite(aug_past).all()
    assert torch.isfinite(aug_current).all()
    assert torch.isfinite(aug_future).all()


def _run_tau(
    inputs: dict[str, torch.Tensor],
    ego_future: torch.Tensor,
    *,
    seed: int,
    ego_past_noise_std: float = 0.1,
    tau_min_s: float = -0.5,
    tau_max_s: float = -0.5,
    neighbors_future: torch.Tensor | None = None,
):
    torch.manual_seed(seed)
    aug = StatePerturbationAtTau(
        augment_prob=1.0,
        num_refine=20,
        device="cpu",
        ego_past_noise_std=ego_past_noise_std,
        use_smoothing_future_trajectory=False,
        tau_min_s=tau_min_s,
        tau_max_s=tau_max_s,
    )
    if neighbors_future is None:
        return aug, *aug.augment_at_tau(inputs, ego_future)
    out_inputs, out_future, _ = aug(inputs, ego_future, neighbors_future)
    return aug, out_inputs, out_future


def test_scale_identity_when_noise_std_is_zero() -> None:
    inputs, ego_future = _straight_scene(current_xy=(3.0, -2.0))
    original_past = inputs["ego_agent_past"].clone()
    aug, flag, aug_current, aug_past, aug_future = _run_tau(
        inputs, ego_future, seed=0, ego_past_noise_std=0.0
    )
    assert flag.tolist() == [True]
    assert aug.last_tau_info["history_scale"] == pytest.approx(1.0)
    # Far-past (before the left bridge) must match the unscaled original.
    left_idx = int(aug.last_tau_info["left_idx"])
    assert torch.allclose(aug_past[0, : left_idx + 1, :2], original_past[0, : left_idx + 1, :2])
    assert _join_error_m(aug_past, aug_current) < 1e-5
    assert torch.isfinite(aug_future).all()


def test_nonzero_current_pose_join_and_speed_continuity() -> None:
    inputs, ego_future = _straight_scene(current_xy=(12.0, -4.0), speed=6.0)
    aug, flag, aug_current, aug_past, aug_future = _run_tau(inputs, ego_future, seed=11)
    assert flag.tolist() == [True]
    join = _join_error_m(aug_past, aug_current)
    jumps = _speed_jump_at_t0(aug_past, aug_current, aug_future)
    assert join < 1e-5
    # Quintic rewrite around t=0 should keep finite-difference speeds close.
    assert jumps["past_vs_future"] < 1.5
    assert jumps["state_vs_future"] < 1.5
    assert torch.isfinite(aug_past).all()
    assert _max_step_m(aug_past, aug_current, aug_future) < 2.0  # 6 m/s * 0.1s * margin


def test_curved_trajectory_remains_smooth_and_finite() -> None:
    inputs, ego_future = _circle_scene(origin=(8.0, 3.0))
    aug, flag, aug_current, aug_past, aug_future = _run_tau(inputs, ego_future, seed=21)
    assert flag.tolist() == [True]
    assert _join_error_m(aug_past, aug_current) < 1e-5
    xy = torch.cat([aug_past[0, :, :2], aug_future[0, :, :2]], dim=0)
    accel = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    max_accel = float(torch.linalg.norm(accel, dim=-1).max().item() / (0.1**2))
    # A 5 m/s turn on R=25 m is ~1 m/s^2 centripetal; quintic offset adds more.
    assert max_accel < 40.0
    assert torch.isfinite(aug_past).all() and torch.isfinite(aug_future).all()
    heading = torch.atan2(aug_past[0, :, 3], aug_past[0, :, 2])
    dpsi = torch.atan2(torch.sin(heading[1:] - heading[:-1]), torch.cos(heading[1:] - heading[:-1]))
    assert float(dpsi.abs().max().item()) < 0.8


def test_rejected_bridge_does_not_mutate_inputs() -> None:
    inputs, ego_future = _straight_scene(current_xy=(5.0, 1.0))
    original_past = inputs["ego_agent_past"].clone()
    original_current = inputs["ego_current_state"].clone()
    aug = StatePerturbationAtTau(
        augment_prob=1.0,
        num_refine=20,
        device="cpu",
        ego_past_noise_std=0.1,
        use_smoothing_future_trajectory=False,
    )

    def always_fail(self, **kwargs):
        return False

    aug._augment_single_at_tau = types.MethodType(always_fail, aug)
    torch.manual_seed(3)
    flag, _, _, _ = aug.augment_at_tau(inputs, ego_future)
    assert flag.tolist() == [False]
    assert torch.equal(inputs["ego_agent_past"], original_past)
    assert torch.equal(inputs["ego_current_state"], original_current)


def test_zero_padded_4d_past_rows_stay_zero() -> None:
    """All-zero 4D padding must survive homothety about a nonzero current pose."""
    inputs, _ = _straight_scene(current_xy=(2.0, -1.0), past_len=5)
    inputs["ego_agent_past"][0, :2] = 0.0
    past = inputs["ego_agent_past"].clone()
    current = inputs["ego_current_state"].clone()
    original_real = past[0, 2:, :2].clone()
    StatePerturbationAtTau._scale_history_about_current(
        ego_current=current,
        ego_past=past,
        aug_flag=torch.tensor([True]),
        scale_by_batch=torch.tensor([1.2]),
    )
    padded = past[0, :2]
    assert torch.equal(padded, torch.zeros_like(padded))
    zero_mask = torch.sum(torch.ne(padded, 0), dim=-1) == 0
    assert bool(zero_mask.all().item())
    assert torch.allclose(past[0, -1, :2], current[0, :2])
    assert not torch.allclose(past[0, 2:, :2], original_real)


def test_heading_converted_origin_row_is_not_treated_as_padding() -> None:
    """[0, 0, 1, 0] is an ego-centric origin pose, not a 4D pad row."""
    ego_current = torch.tensor([[2.0, -1.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    ego_past = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, -1.0, 1.0, 0.0],
                [2.0, -1.0, 1.0, 0.0],
            ]
        ]
    )
    StatePerturbationAtTau._scale_history_about_current(
        ego_current=ego_current,
        ego_past=ego_past,
        aug_flag=torch.tensor([True]),
        scale_by_batch=torch.tensor([1.2]),
    )
    # Homothety of (0,0) about (2,-1) with s=1.2 is (-0.4, 0.2); heading stays (1, 0).
    assert torch.allclose(ego_past[0, 0, :2], torch.tensor([-0.4, 0.2]), atol=1e-5)
    assert torch.allclose(ego_past[0, 0, 2:4], torch.tensor([1.0, 0.0]))


def test_random_seeds_are_finite_with_bounded_joins() -> None:
    join_errors = []
    speed_gaps = []
    accepted = 0
    for seed in range(40):
        origin = (float(seed % 5), float((seed % 7) - 3))
        if seed % 2 == 0:
            inputs, ego_future = _straight_scene(current_xy=origin, speed=4.0 + (seed % 5))
        else:
            inputs, ego_future = _circle_scene(origin=origin)
        aug, flag, aug_current, aug_past, aug_future = _run_tau(
            inputs,
            ego_future,
            seed=seed,
            tau_min_s=-1.0,
            tau_max_s=0.0,
        )
        if not bool(flag[0].item()):
            continue
        accepted += 1
        join_errors.append(_join_error_m(aug_past, aug_current))
        speed_gaps.append(
            _speed_jump_at_t0(aug_past, aug_current, aug_future)["past_vs_future"]
        )
        assert torch.isfinite(aug_past).all()
        assert torch.isfinite(aug_current).all()
        assert torch.isfinite(aug_future).all()
        assert torch.isfinite(torch.linalg.norm(aug_current[0, 4:6]))
    assert accepted >= 30
    assert max(join_errors) < 1e-5
    assert sum(speed_gaps) / len(speed_gaps) < 1.0


def test_polyline_tangential_va_matches_backward_stencil() -> None:
    dt = 0.1
    # x = 8t - t^2  (v=8-2t, a=-2), heading 0
    t = torch.arange(5) * dt
    xy = torch.stack([8.0 * t - t**2, torch.zeros(5)], dim=-1)
    heading = torch.zeros(5)
    v, a = polyline_tangential_va(xy, heading, index=4, dt=dt)
    # Backward stencil (x[i]-x[i-1])/dt, not the instantaneous analytic v(t_end).
    expected_v = (xy[4, 0] - xy[3, 0]) / dt
    expected_a = (xy[4, 0] - 2 * xy[3, 0] + xy[2, 0]) / dt**2
    assert v.item() == pytest.approx(expected_v.item(), abs=1e-5)
    assert a.item() == pytest.approx(expected_a.item(), abs=1e-4)


def test_tau_t0_uses_signed_tangential_accel() -> None:
    """Braking current state must enter the quintic as a<0, not ||(ax, ay)||."""
    inputs, ego_future = _straight_scene(speed=8.0, current_xy=(0.0, 0.0))
    inputs["ego_current_state"][0, 6] = -2.0  # ax braking
    heading = torch.zeros(())
    v, a = tangential_va(
        inputs["ego_current_state"][0, 4:6],
        inputs["ego_current_state"][0, 6:8],
        heading,
    )
    assert a.item() == pytest.approx(-2.0)
    assert torch.linalg.norm(inputs["ego_current_state"][0, 6:8]).item() == pytest.approx(2.0)

    torch.manual_seed(0)
    aug = StatePerturbationAtTau(
        augment_prob=1.0,
        num_refine=20,
        device="cpu",
        ego_past_noise_std=0.0,
        use_smoothing_future_trajectory=False,
        tau_min_s=0.0,
        tau_max_s=0.0,
    )
    flag, _, _, _ = aug.augment_at_tau(inputs, ego_future)
    assert flag.tolist() == [True]
    assert aug.last_tau_info is not None
    # accel_off is in [-0.2, 0.2]; a_gt is the signed -2, not +2.
    assert aug.last_tau_info["a_gt_mps2"] == pytest.approx(-2.0, abs=0.05)
    assert aug.last_tau_info["a_tau_mps2"] < 0.0
