import numpy as np
import torch

from diffusion_planner.utils.data_augmentation_geom import (
    TIME_INTERVAL,
    _cross2d,
    _rect_corners,
    _sat_signed_distance,
    _segments_intersect_rect,
    heading_from_cos_sin,
    heading_transform,
    polyline_tangential_va,
    project_onto_heading,
    rotate_xy_by_heading,
    tangential_va,
    vector_transform,
)
from diffusion_planner.utils.data_augmentation_scene import (
    centric_transform as scene_centric_transform,
    check_aug_validity,
    get_transform_matrix_batch,
)

# Re-export geometry helpers for existing imports from this module.
__all__ = [
    "TIME_INTERVAL",
    "StatePerturbation",
    "_cross2d",
    "_rect_corners",
    "_sat_signed_distance",
    "_segments_intersect_rect",
    "heading_from_cos_sin",
    "heading_transform",
    "polyline_tangential_va",
    "project_onto_heading",
    "rotate_xy_by_heading",
    "tangential_va",
    "vector_transform",
]


class StatePerturbation:
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory that
    satisfies polynomial constraints.
    """

    def __init__(
        self,
        augment_prob: float,
        num_refine: int,
        device: torch.device | str,
        ego_past_noise_std: float,
        use_smoothing_future_trajectory: bool,
    ) -> None:
        """
        Initialize the augmentor,
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param num_refine: number of refinement steps for quintic interpolation
        :param device: torch device
        :param ego_past_noise_std: std of noise applied to ego past trajectory
        :param use_smoothing_future_trajectory: whether to apply smoothing to future trajectory
        """
        self._augment_prob = augment_prob
        self._device = torch.device(device)
        self._ego_past_noise_std = ego_past_noise_std
        self._use_smoothing_future_trajectory = use_smoothing_future_trajectory
        # Subclasses that rewrite ego past into the same pre-centric frame as the
        # new current pose should set this True so centric_transform rotates past.
        self._transform_ego_past = False
        lo = [0.0, -0.75, -0.2, -1, -0.5, -0.2, -0.1, 0.0, 0.0]
        hi = [0.0, +0.75, +0.2, +1, +0.5, +0.2, +0.1, 0.0, 0.0]
        # Shape (9,) so that len(self._low) is the number of perturbation dims;
        # the previous (1, 9) shape made torch.rand(B, len(self._low)) sample a
        # single scalar per sample, perfectly correlating all 9 perturbations.
        self._low = torch.tensor(lo).to(self._device)
        self._high = torch.tensor(hi).to(self._device)

        self.num_refine = num_refine
        self.time_interval = TIME_INTERVAL

        REFINE_HORIZON = num_refine * TIME_INTERVAL

        T = REFINE_HORIZON + TIME_INTERVAL
        self.coeff_matrix = torch.linalg.inv(
            torch.tensor(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0],
                    [1, T, T**2, T**3, T**4, T**5],
                    [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                    [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
                ],
                device=device,
                dtype=torch.float32,
            )
        )
        self.t_matrix = torch.pow(
            torch.linspace(TIME_INTERVAL, REFINE_HORIZON, num_refine).unsqueeze(1),
            torch.arange(6).unsqueeze(0),
        ).to(device=device)  # shape (B, N+1)

    def __call__(self, inputs, ego_future, neighbors_future):
        aug_flag, aug_ego_current_state = self.augment(inputs)

        # Scale past and current v/a BEFORE interpolating the future. The quintic
        # uses current velocity/acceleration as its start BC, so scaling afterwards
        # left the state reporting a speed the target does not start with.
        B_aug = aug_flag.sum().item()
        if B_aug > 0:
            W = self._ego_past_noise_std
            scale = torch.normal(mean=1.0, std=W, size=(B_aug, 1, 1)).to(
                inputs["ego_agent_past"].device
            )
            scale = torch.clamp(scale, 1.0 - 2 * W, 1.0 + 2 * W)

            ego_past_aug = inputs["ego_agent_past"][aug_flag].clone()
            ego_past_aug[..., :2] = ego_past_aug[..., :2] * scale
            inputs["ego_agent_past"][aug_flag] = ego_past_aug

            scale_1d = scale.squeeze(-1)  # (B_aug, 1)
            aug_ego_current_state[aug_flag, 4:6] *= scale_1d  # vx, vy
            aug_ego_current_state[aug_flag, 6:8] *= scale_1d  # ax, ay

        interpolated_ego_future = self.interpolation_future_trajectory(
            aug_ego_current_state, ego_future
        )

        inputs["ego_current_state"][aug_flag] = aug_ego_current_state[aug_flag]
        ego_future[aug_flag] = interpolated_ego_future[aug_flag]

        return self.centric_transform(inputs, ego_future, neighbors_future)

    def augment(self, inputs):
        # Only aug current state
        ego_current_state = inputs["ego_current_state"].clone()
        wheel_base = inputs["ego_shape"][:, 0]  # (B,)

        B = ego_current_state.shape[0]
        aug_flag = (torch.rand(B) < self._augment_prob).bool().to(self._device) & ~(
            abs(ego_current_state[:, 4]) < 2.0
        )

        random_tensor = torch.rand(B, len(self._low)).to(self._device)
        scaled_random_tensor = self._low + (self._high - self._low) * random_tensor

        new_state = torch.zeros((B, 9), dtype=torch.float32).to(self._device)
        new_state[:, 3:] = ego_current_state[
            :, 4:10
        ]  # x, y, h is 0 because of ego-centric, update vx, vy, ax, ay, steering angle, yaw rate
        new_state = new_state + scaled_random_tensor
        new_state[:, 3] = torch.max(new_state[:, 3], torch.tensor(0.0, device=new_state.device))
        new_state[:, -1] = torch.clip(new_state[:, -1], -0.85, 0.85)

        ego_current_state[:, :2] = new_state[:, :2]
        ego_current_state[:, 2] = torch.cos(new_state[:, 2])
        ego_current_state[:, 3] = torch.sin(new_state[:, 2])
        ego_current_state[:, 4:8] = new_state[:, 3:7]
        ego_current_state[:, 8:10] = new_state[:, -2:]  # steering angle, yaw rate

        # update steering angle and yaw rate
        cur_velocity = ego_current_state[:, 4]
        yaw_rate = ego_current_state[:, 9]

        steering_angle = torch.zeros_like(cur_velocity)
        new_yaw_rate = torch.zeros_like(yaw_rate)

        mask = torch.abs(cur_velocity) < 0.2
        not_mask = ~mask
        steering_angle[not_mask] = torch.atan(
            yaw_rate[not_mask] * wheel_base[not_mask] / torch.abs(cur_velocity[not_mask])
        )
        steering_angle[not_mask] = torch.clamp(
            steering_angle[not_mask], -2 / 3 * np.pi, 2 / 3 * np.pi
        )
        new_yaw_rate[not_mask] = yaw_rate[not_mask]

        ego_current_state[:, 8] = steering_angle
        ego_current_state[:, 9] = new_yaw_rate

        # ay is centripetal: vx * yaw_rate. Perturbing vx while carrying ay over
        # breaks that by dvx * yaw_rate. Rebuild from the perturbed speed and keep
        # the sampled ay noise on top (same treatment as steering_angle above).
        ego_current_state[:, 7] = (
            ego_current_state[:, 4] * new_yaw_rate + scaled_random_tensor[:, 6]
        )

        # Body-frame v/a would pick up sideslip after centric_transform rotates by
        # R(-delta_heading). Pre-rotate by R(+heading) so the round trip is identity.
        cos_h = ego_current_state[:, 2]
        sin_h = ego_current_state[:, 3]
        ego_current_state[:, 4:6] = rotate_xy_by_heading(
            ego_current_state[:, 4:6], cos_h, sin_h
        )
        ego_current_state[:, 6:8] = rotate_xy_by_heading(
            ego_current_state[:, 6:8], cos_h, sin_h
        )

        # Discard augmentations that cause collisions
        collision = self._check_aug_validity(ego_current_state, inputs)
        aug_flag = aug_flag & ~collision

        return aug_flag, ego_current_state

    def _check_aug_validity(self, aug_ego_state: torch.Tensor, inputs: dict) -> torch.Tensor:
        return check_aug_validity(aug_ego_state, inputs)

    def normalize_angle(self, angle: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_transform_matrix_batch(self, cur_state):
        return get_transform_matrix_batch(cur_state)

    def centric_transform(
        self,
        inputs: torch.Tensor,
        ego_future: torch.Tensor,
        neighbors_future: torch.Tensor,
    ):
        return scene_centric_transform(
            inputs,
            ego_future,
            neighbors_future,
            use_smoothing_future_trajectory=self._use_smoothing_future_trajectory,
            transform_ego_past=self._transform_ego_past,
        )

    def interpolation_future_trajectory(self, aug_current_state, ego_future, keep_remaining=True):
        """
        refine future trajectory with quintic Hermite interpolation

        Args:
            aug_current_state: (B, 16) current state of the ego vehicle after augmentation
            ego_future:        (B, T, 3) future trajectory of the ego vehicle
            keep_remaining:    If True, keep the remaining trajectory after P frames (default: True)

        Returns:
            ego_future: refined future trajectory of the ego vehicle
        """

        P = self.num_refine
        dt = self.time_interval
        B = aug_current_state.shape[0]
        M_t = self.t_matrix.unsqueeze(0).expand(B, -1, -1)
        A = self.coeff_matrix.unsqueeze(0).expand(B, -1, -1)

        # theta0 is the heading of the (perturbed) current state, not the chord
        # to a point 1 s ahead. v0/a0 are signed tangential components: torch.norm
        # dropped braking sign and double-counted centripetal v*omega (already in
        # the quintic via -v0*sin(theta0)*omega0).
        x0 = aug_current_state[:, 0]
        y0 = aug_current_state[:, 1]
        theta0 = heading_from_cos_sin(aug_current_state[:, 2], aug_current_state[:, 3])
        omega0 = aug_current_state[:, 9]
        v0, a0 = tangential_va(
            aug_current_state[:, 4:6], aug_current_state[:, 6:8], theta0
        )

        xT = ego_future[:, P, 0]
        yT = ego_future[:, P, 1]
        thetaT = ego_future[:, P, 2]
        omegaT = self.normalize_angle(ego_future[:, P, 2] - ego_future[:, P - 1, 2]) / dt
        d1 = (ego_future[:, P, :2] - ego_future[:, P - 1, :2]) / dt
        d2 = (
            ego_future[:, P, :2] - 2 * ego_future[:, P - 1, :2] + ego_future[:, P - 2, :2]
        ) / dt**2
        vT = project_onto_heading(d1, thetaT)
        aT = project_onto_heading(d2, thetaT)

        # Boundary conditions
        sx = torch.stack(
            [
                x0,
                v0 * torch.cos(theta0),
                a0 * torch.cos(theta0) - v0 * torch.sin(theta0) * omega0,
                xT,
                vT * torch.cos(thetaT),
                aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT,
            ],
            dim=-1,
        )

        sy = torch.stack(
            [
                y0,
                v0 * torch.sin(theta0),
                a0 * torch.sin(theta0) + v0 * torch.cos(theta0) * omega0,
                yT,
                vT * torch.sin(thetaT),
                aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT,
            ],
            dim=-1,
        )

        ax = A @ sx[:, :, None]  # B, 6, 1
        ay = A @ sy[:, :, None]  # B, 6, 1

        traj_x = M_t @ ax
        traj_y = M_t @ ay
        traj_heading = torch.cat(
            [
                torch.atan2(
                    traj_y[:, :1, 0] - y0.unsqueeze(-1), traj_x[:, :1, 0] - x0.unsqueeze(-1)
                ),
                torch.atan2(
                    traj_y[:, 1:, 0] - traj_y[:, :-1, 0], traj_x[:, 1:, 0] - traj_x[:, :-1, 0]
                ),
            ],
            dim=1,
        )

        interpolated = torch.cat([traj_x, traj_y, traj_heading[..., None]], axis=-1)

        if keep_remaining and ego_future.shape[1] > P:
            return torch.concatenate([interpolated, ego_future[:, P:, :]], axis=1)
        else:
            return interpolated


from diffusion_planner.utils.data_augmentation_tau import TauOffsetBump

# Backward compatibility for legacy scripts importing StatePerturbationAtTau.
StatePerturbationAtTau = TauOffsetBump
