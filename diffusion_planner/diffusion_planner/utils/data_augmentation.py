import numpy as np
import torch

from diffusion_planner.utils.unicycle_accel_curvature import smoothing_future_trajectory

TIME_INTERVAL = 0.1


def vector_transform(vector, transform_mat, bias=None):
    """
    vector: (B, ..., 2)
    transform_mat: (B, 2, 2)
    bias: (B, ..., 2)
    """
    shape = vector.shape
    B = vector.shape[0]
    nexpand = vector.ndim - 2
    if bias is not None:
        vector = vector - bias.reshape(B, *([1] * nexpand), -1)
    vector = vector.reshape(B, -1, 2).permute(0, 2, 1)  # (B, 2, N1 * N2 ...)
    return torch.bmm(transform_mat, vector).permute(0, 2, 1).reshape(*shape)  # (B, ..., 2)


def heading_transform(heading, transform_mat):
    """
    heading: (B, ...)
    transform_mat: (B, 2, 2)
    """
    B = heading.shape[0]
    shape = heading.shape
    heading = heading.reshape(B, -1)
    transform_mat = transform_mat.reshape(B, 1, 2, 2)
    return torch.atan2(
        torch.cos(heading) * transform_mat[..., 1, 0]
        + torch.sin(heading) * transform_mat[..., 1, 1],
        torch.cos(heading) * transform_mat[..., 0, 0]
        + torch.sin(heading) * transform_mat[..., 0, 1],
    ).reshape(*shape)


def heading_from_cos_sin(cos_h: torch.Tensor, sin_h: torch.Tensor) -> torch.Tensor:
    """Heading (rad) from a (cos, sin) pair. Broadcasts like ``torch.atan2``."""
    return torch.atan2(sin_h, cos_h)


def project_onto_heading(xy: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Signed projection of a 2-vector onto heading.

    Args:
        xy: (..., 2) velocity, acceleration, or finite-difference vector.
        heading: (...) radians, broadcastable with ``xy`` without the last dim.

    Returns:
        (...) signed scalar along ``heading``. Positive is forward.
    """
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    return xy[..., 0] * cos_h + xy[..., 1] * sin_h


def tangential_va(
    velocity_xy: torch.Tensor,
    acceleration_xy: torch.Tensor,
    heading: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed tangential speed and acceleration along ``heading``.

    Unlike ``torch.norm``, this keeps the sign of braking and does not fold the
    centripetal (lateral) component into longitudinal acceleration. The quintic
    already accounts for centripetal acceleration via its ``v * omega`` terms.
    """
    return (
        project_onto_heading(velocity_xy, heading),
        project_onto_heading(acceleration_xy, heading),
    )


def polyline_tangential_va(
    xy: torch.Tensor,
    heading: torch.Tensor,
    index: int,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference velocity/acceleration at ``index``, projected on heading.

    Matches ``interpolation_future_trajectory``: backward first/second differences
    when two samples exist behind ``index``, otherwise forward differences.
    """
    n = int(xy.shape[0])
    i = int(index)
    zeros = xy.new_zeros(xy.shape[-1])
    if n < 2:
        heading_i = heading[i] if heading.ndim > 0 and heading.numel() > 1 else heading
        return project_onto_heading(zeros, heading_i), project_onto_heading(zeros, heading_i)

    if i >= 2:
        vel_xy = (xy[i] - xy[i - 1]) / dt
        acc_xy = (xy[i] - 2 * xy[i - 1] + xy[i - 2]) / dt**2
    elif i <= 0:
        vel_xy = (xy[1] - xy[0]) / dt
        acc_xy = (xy[2] - 2 * xy[1] + xy[0]) / dt**2 if n >= 3 else zeros
    else:
        vel_xy = (xy[i] - xy[i - 1]) / dt
        acc_xy = (xy[i + 1] - 2 * xy[i] + xy[i - 1]) / dt**2 if n >= 3 else zeros

    heading_i = heading[i] if heading.ndim > 0 and heading.numel() > 1 else heading
    return project_onto_heading(vel_xy, heading_i), project_onto_heading(acc_xy, heading_i)


def rotate_xy_by_heading(
    xy: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
) -> torch.Tensor:
    """Rotate ``xy`` (..., 2) by heading ``(cos_h, sin_h)``.

    Used to pre-rotate body-frame velocity/acceleration so ``centric_transform``'s
    ``R(-heading)`` restores the original body-frame components (no sideslip).
    """
    lon = xy[..., 0]
    lat = xy[..., 1]
    return torch.stack([cos_h * lon - sin_h * lat, sin_h * lon + cos_h * lat], dim=-1)


def _cross2d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """2D cross product along the last dimension: u × v = u.x*v.y - u.y*v.x"""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _rect_corners(rect: torch.Tensor) -> torch.Tensor:
    """
    rect: [B, 6] — (x, y, cos_h, sin_h, length, width)
    Returns [B, 4, 2] corner points.
    """
    B = rect.shape[0]
    xy, cos_h, sin_h, lw = rect[:, :2], rect[:, 2], rect[:, 3], rect[:, 4:]
    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=1).reshape(B, 2, 2)
    signs = torch.tensor([[1.0, 1], [-1, 1], [-1, -1], [1, -1]], device=lw.device)
    local = torch.einsum("bj,ij->bij", lw / 2, signs)  # [B, 4, 2]
    local = torch.einsum("bij,bkj->bik", local, rot)  # [B, 4, 2]
    return xy[:, None, :] + local


def _sat_signed_distance(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """
    SAT signed distance between two rectangles.
    c1, c2: [B, 4, 2] corner points
    Returns [B] — negative means overlap.
    """
    nv = torch.stack(
        [c1[:, 0] - c1[:, 1], c1[:, 1] - c1[:, 2], c2[:, 0] - c2[:, 1], c2[:, 1] - c2[:, 2]],
        dim=1,
    )  # [B, 4, 2]
    nv = nv / torch.norm(nv, dim=2, keepdim=True).clamp(min=1e-6)
    p1 = torch.einsum("bij,bkj->bik", nv, c1)  # [B, 4, 4]
    p2 = torch.einsum("bij,bkj->bik", nv, c2)
    overlap = torch.cat(
        [p1.min(2).values - p2.max(2).values, p2.min(2).values - p1.max(2).values],
        dim=1,
    )  # [B, 8]
    is_overlap = (overlap < 0).all(dim=1)
    pos = torch.where(overlap < 0, torch.full_like(overlap, 1e5), overlap)
    return torch.where(is_overlap, overlap.max(1).values, pos.min(1).values)


def _segments_intersect_rect(
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    rect_corners: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Returns [B] bool — True if any valid segment touches the rectangle.

    seg_start, seg_end: [B, N, 2]
    rect_corners:       [B, 4, 2]
    valid:              [B, N] bool — True for valid segments
    """
    hit = torch.zeros(seg_start.shape[:2], dtype=torch.bool, device=seg_start.device)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # Proper segment–edge crossing: both pairs straddle each other's line
    for i, j in edges:
        C = rect_corners[:, i, :].unsqueeze(1)  # [B, 1, 2]
        D = rect_corners[:, j, :].unsqueeze(1)  # [B, 1, 2]
        AB = seg_end - seg_start  # [B, N, 2]
        CD = D - C  # [B, 1, 2]
        hit = hit | (
            (_cross2d(AB, C - seg_start) * _cross2d(AB, D - seg_start) < 0)
            & (_cross2d(CD, seg_start - C) * _cross2d(CD, seg_end - C) < 0)
        )

    # Endpoint inside polygon: all edge cross products share the same sign
    for pt in (seg_start, seg_end):
        crosses = torch.stack(
            [
                _cross2d(
                    (rect_corners[:, j, :] - rect_corners[:, i, :]).unsqueeze(1),
                    pt - rect_corners[:, i, :].unsqueeze(1),
                )
                for i, j in edges
            ],
            dim=-1,
        )  # [B, N, 4]
        hit = hit | (crosses > 0).all(-1) | (crosses < 0).all(-1)

    if valid is not None:
        hit = hit & valid
    return hit.any(dim=1)  # [B]


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
        """
        Returns [B] bool — True where the augmented ego position is invalid.

        Invalid conditions:
          1. Ego polygon overlaps with a neighbour agent polygon.
          2. Ego polygon intersects a road-border segment from ``line_strings``.

        Lane left/right boundaries are intentionally not used: they frequently
        intersect the GT ego footprint (adjacent/merging polylines), which made
        the previous check reject a large fraction of otherwise valid samples.
        Road borders (``line_strings[..., 3]`` one-hot) are the drivable-area
        edge and match the training road-border penalty representation.
        """
        B = aug_ego_state.shape[0]
        device = aug_ego_state.device
        dtype = aug_ego_state.dtype

        # ego_shape: [B, 3] = (wheelbase, length, width)
        ego_shape = inputs["ego_shape"].to(device=device, dtype=dtype)
        ego_length = ego_shape[:, 1:2]  # [B, 1]
        ego_width = ego_shape[:, 2:3]  # [B, 1]

        ego_rect = torch.cat(
            [aug_ego_state[:, :4], ego_length, ego_width],
            dim=-1,
        )  # [B, 6]
        ego_corners = _rect_corners(ego_rect)  # [B, 4, 2]

        collision = torch.zeros(B, dtype=torch.bool, device=device)

        # ── 1. Neighbour agent polygon collision ──────────────────────────────
        if "neighbor_agents_past" in inputs:
            nbr = inputs["neighbor_agents_past"][:, :, -1, :]  # [B, N, 11]
            N = nbr.shape[1]
            valid = torch.sum(torch.ne(nbr[:, :, :4], 0), dim=-1) > 0  # [B, N]
            if valid.any():
                # neighbor_agents_past layout: x,y,cos,sin (0:4), width (6), length (7)
                nbr_rect = torch.cat(
                    [nbr[:, :, :4], nbr[:, :, 7:8], nbr[:, :, 6:7]], dim=-1
                )  # [B, N, 6]  — (x,y,cos,sin,length,width)
                dists = _sat_signed_distance(
                    _rect_corners(ego_rect.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 6)),
                    _rect_corners(nbr_rect.reshape(B * N, 6)),
                ).reshape(B, N)
                collision = collision | ((dists < 0) & valid).any(dim=1)

        # ── 2. Road-border segment collision ─────────────────────────────────
        # line_strings layout: [B, N_ls, P, D] with D>=4 → (x, y, stop_line, road_border)
        if "line_strings" in inputs:
            line_strings = inputs["line_strings"]
            if line_strings.shape[-1] >= 4:
                pts = line_strings[..., :2]  # [B, N_ls, P, 2]
                # A polyline is a road border if any point carries the flag.
                is_road_border = (line_strings[..., 3] > 0.5).any(dim=-1)  # [B, N_ls]
                # Point valid when xy is non-trivial.
                point_valid = torch.norm(pts, dim=-1) > 1e-6  # [B, N_ls, P]
                # Segment valid only on road-border polylines with two valid endpoints.
                seg_valid = (
                    point_valid[:, :, :-1]
                    & point_valid[:, :, 1:]
                    & is_road_border[:, :, None]
                )  # [B, N_ls, P-1]

                seg_start = pts[:, :, :-1, :].reshape(B, -1, 2)
                seg_end = pts[:, :, 1:, :].reshape(B, -1, 2)
                seg_valid_flat = seg_valid.reshape(B, -1)

                if seg_valid_flat.any():
                    collision = collision | _segments_intersect_rect(
                        seg_start,
                        seg_end,
                        ego_corners,
                        seg_valid_flat,
                    )

        return collision

    def normalize_angle(self, angle: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_transform_matrix_batch(self, cur_state):
        processed_input = torch.column_stack(
            (
                cur_state[:, 2],  # cos
                cur_state[:, 3],  # sin
            )
        )

        reshaping_tensor = torch.tensor(
            [
                [1, 0, 0, 1],
                [0, 1, -1, 0],
            ],
            dtype=torch.float32,
        ).to(processed_input.device)
        return (processed_input @ reshaping_tensor).reshape(-1, 2, 2)

    def centric_transform(
        self,
        inputs: torch.Tensor,
        ego_future: torch.Tensor,
        neighbors_future: torch.Tensor,
    ):
        cur_state = inputs["ego_current_state"].clone()
        center_xy = cur_state[:, :2]
        transform_matrix = self.get_transform_matrix_batch(cur_state)

        # ego xy
        inputs["ego_current_state"][..., :2] = vector_transform(
            inputs["ego_current_state"][..., :2], transform_matrix, center_xy
        )
        # ego cos sin
        inputs["ego_current_state"][..., 2:4] = vector_transform(
            inputs["ego_current_state"][..., 2:4], transform_matrix
        )
        # ego vx, vy
        inputs["ego_current_state"][..., 4:6] = vector_transform(
            inputs["ego_current_state"][..., 4:6], transform_matrix
        )
        # ego ax, ay
        inputs["ego_current_state"][..., 6:8] = vector_transform(
            inputs["ego_current_state"][..., 6:8], transform_matrix
        )

        # ego future xy
        ego_future[..., :2] = vector_transform(ego_future[..., :2], transform_matrix, center_xy)
        ego_future[..., 2] = heading_transform(ego_future[..., 2], transform_matrix)

        # ego past — only when past was rewritten into the perturbed frame
        if self._transform_ego_past:
            ego_past_mask = (
                torch.sum(torch.ne(inputs["ego_agent_past"][..., :4], 0), dim=-1) == 0
            )
            inputs["ego_agent_past"][..., :2] = vector_transform(
                inputs["ego_agent_past"][..., :2], transform_matrix, center_xy
            )
            inputs["ego_agent_past"][..., 2:4] = vector_transform(
                inputs["ego_agent_past"][..., 2:4], transform_matrix
            )
            inputs["ego_agent_past"][ego_past_mask] = 0.0

        ego_past4d = inputs["ego_agent_past"]
        ego_future4d = torch.cat(
            [
                ego_future[..., :2],  # x, y
                torch.cos(ego_future[..., 2:3]),  # cos
                torch.sin(ego_future[..., 2:3]),  # sin
            ],
            dim=-1,
        )

        if self._use_smoothing_future_trajectory:
            ego_future4d = smoothing_future_trajectory(
                ego_past4d, inputs["ego_current_state"], ego_future4d
            )

        ego_future = torch.cat(
            [
                ego_future4d[..., :2],  # x, y
                torch.atan2(ego_future4d[..., 3], ego_future4d[..., 2]).unsqueeze(
                    -1
                ),  # heading from cos, sin
            ],
            dim=-1,
        )
        inputs["ego_agent_future"] = ego_future

        # goal pose (x, y, cos, sin)
        # Validity is decided from the position only: heading_to_cos_sin turns an
        # all-zero (x, y, heading) goal into (0, 0, 1, 0), so a full-width zero test
        # never fires and an absent goal would survive as a goal at the ego itself.
        mask = torch.sum(torch.ne(inputs["goal_pose"][..., :2], 0), dim=-1) == 0
        inputs["goal_pose"][..., :2] = vector_transform(
            inputs["goal_pose"][..., :2], transform_matrix, center_xy
        )
        inputs["goal_pose"][..., 2:4] = vector_transform(
            inputs["goal_pose"][..., 2:4], transform_matrix
        )
        inputs["goal_pose"][mask] = 0.0

        # neighbor past xy
        mask = torch.sum(torch.ne(inputs["neighbor_agents_past"][..., :6], 0), dim=-1) == 0
        inputs["neighbor_agents_past"][..., :2] = vector_transform(
            inputs["neighbor_agents_past"][..., :2], transform_matrix, center_xy
        )
        # neighbor past cos sin
        inputs["neighbor_agents_past"][..., 2:4] = vector_transform(
            inputs["neighbor_agents_past"][..., 2:4], transform_matrix
        )
        # neighbor past vx, vy
        inputs["neighbor_agents_past"][..., 4:6] = vector_transform(
            inputs["neighbor_agents_past"][..., 4:6], transform_matrix
        )
        inputs["neighbor_agents_past"][mask] = 0.0

        # neighbor future xy
        mask = torch.sum(torch.ne(neighbors_future[..., :2], 0), dim=-1) == 0
        neighbors_future[..., :2] = vector_transform(
            neighbors_future[..., :2], transform_matrix, center_xy
        )
        if neighbors_future.shape[-1] == 4:
            # Canonical [x, y, cos, sin]: rotate the heading unit vector.
            neighbors_future[..., 2:4] = vector_transform(
                neighbors_future[..., 2:4], transform_matrix
            )
        else:
            neighbors_future[..., 2] = heading_transform(neighbors_future[..., 2], transform_matrix)
        neighbors_future[mask] = 0.0

        # lanes
        mask = torch.sum(torch.ne(inputs["lanes"][..., :8], 0), dim=-1) == 0
        inputs["lanes"][..., :2] = vector_transform(
            inputs["lanes"][..., :2], transform_matrix, center_xy
        )
        inputs["lanes"][..., 2:4] = vector_transform(inputs["lanes"][..., 2:4], transform_matrix)
        inputs["lanes"][..., 4:6] = vector_transform(inputs["lanes"][..., 4:6], transform_matrix)
        inputs["lanes"][..., 6:8] = vector_transform(inputs["lanes"][..., 6:8], transform_matrix)
        inputs["lanes"][mask] = 0.0

        # route_lanes
        mask = torch.sum(torch.ne(inputs["route_lanes"][..., :8], 0), dim=-1) == 0
        inputs["route_lanes"][..., :2] = vector_transform(
            inputs["route_lanes"][..., :2], transform_matrix, center_xy
        )
        inputs["route_lanes"][..., 2:4] = vector_transform(
            inputs["route_lanes"][..., 2:4], transform_matrix
        )
        inputs["route_lanes"][..., 4:6] = vector_transform(
            inputs["route_lanes"][..., 4:6], transform_matrix
        )
        inputs["route_lanes"][..., 6:8] = vector_transform(
            inputs["route_lanes"][..., 6:8], transform_matrix
        )
        inputs["route_lanes"][mask] = 0.0

        # polygons
        mask = torch.sum(torch.ne(inputs["polygons"], 0), dim=-1) == 0
        inputs["polygons"][..., :2] = vector_transform(
            inputs["polygons"][..., :2], transform_matrix, center_xy
        )
        inputs["polygons"][mask] = 0.0

        # line_strings
        mask = torch.sum(torch.ne(inputs["line_strings"], 0), dim=-1) == 0
        inputs["line_strings"][..., :2] = vector_transform(
            inputs["line_strings"][..., :2], transform_matrix, center_xy
        )
        inputs["line_strings"][mask] = 0.0

        # static objects xy
        mask = torch.sum(torch.ne(inputs["static_objects"][..., :10], 0), dim=-1) == 0
        inputs["static_objects"][..., :2] = vector_transform(
            inputs["static_objects"][..., :2], transform_matrix, center_xy
        )
        # static objects cos sin
        inputs["static_objects"][..., 2:4] = vector_transform(
            inputs["static_objects"][..., 2:4], transform_matrix
        )
        inputs["static_objects"][mask] = 0.0

        return inputs, ego_future, neighbors_future

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


def _quintic_coeff_matrix(duration_s: float, device, dtype) -> torch.Tensor:
    """6x6 matrix mapping boundary vector [p0,p'0,p''0,pT,p'T,p''T] to polynomial coeffs."""
    T = float(duration_s)
    if T <= 0.0:
        raise ValueError("duration_s must be positive.")
    return torch.linalg.inv(
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
            dtype=dtype,
        )
    )


class StatePerturbationAtTau(StatePerturbation):
    """Perturb at a random time τ ∈ [tau_min, tau_max] and reconnect with two quintics.

    Pose + kinematics recovery (extends pose-first v1):

    - Sample τ ~ Uniform[tau_min_s, tau_max_s] (default [-1, 0]).
    - At τ apply:
        * lateral / heading offset (``_low``/``_high`` indices 1, 2),
        * speed / accel offset (indices 3, 5 — same ranges as parent vx / ax),
        * yaw-rate kept from GT at τ (parent also leaves ω unperturbed; steering
          is re-derived from ω and the new speed, matching parent ``augment``).
    - Quintic bridge on each side of τ over ``num_refine * time_interval`` so
      pose **and** speed/steer recover continuously toward GT at the bridge ends.
    - Apply the optional history-speed scale before building either bridge,
      anchored at the current pose rather than at the global origin.
    - Re-derive ``ego_current_state`` at t=0 from the rewritten trajectory.
    - Collision check + centric transform (with ego-past transform enabled).
    """

    def __init__(
        self,
        augment_prob: float,
        num_refine: int,
        device: torch.device | str,
        ego_past_noise_std: float,
        use_smoothing_future_trajectory: bool,
        tau_min_s: float = -1.0,
        tau_max_s: float = 0.0,
    ) -> None:
        super().__init__(
            augment_prob=augment_prob,
            num_refine=num_refine,
            device=device,
            ego_past_noise_std=ego_past_noise_std,
            use_smoothing_future_trajectory=use_smoothing_future_trajectory,
        )
        if tau_min_s > tau_max_s:
            raise ValueError("tau_min_s must be <= tau_max_s.")
        self._tau_min_s = float(tau_min_s)
        self._tau_max_s = float(tau_max_s)
        self._transform_ego_past = True
        # Unlike the parent path, this scale is applied before the tau bridges are built.
        self._ego_past_noise_std = float(ego_past_noise_std)
        # Last successful sample metadata for visualization / debugging.
        self.last_tau_info: dict | None = None

    @property
    def refine_horizon_s(self) -> float:
        """Same duration as the parent quintic refine window (``num_refine * dt``)."""
        return float(self.num_refine * self.time_interval)

    def __call__(self, inputs, ego_future, neighbors_future):
        aug_flag, aug_current, aug_past, aug_future = self.augment_at_tau(inputs, ego_future)

        inputs["ego_current_state"][aug_flag] = aug_current[aug_flag]
        inputs["ego_agent_past"][aug_flag] = aug_past[aug_flag]
        ego_future[aug_flag] = aug_future[aug_flag]

        return self.centric_transform(inputs, ego_future, neighbors_future)

    @staticmethod
    def _scale_history_about_current(
        ego_current: torch.Tensor,
        ego_past: torch.Tensor,
        aug_flag: torch.Tensor,
        scale_by_batch: torch.Tensor,
    ) -> None:
        """Dilate accepted histories about their current pose, in place.

        ``ego_past`` and ``ego_current`` are still in the same pre-centric frame here.
        Anchoring the dilation at the current position preserves
        ``ego_past[-1].xy == ego_current.xy`` even when that position is not the origin.
        The matching velocity/acceleration scale is consumed by the tau bridge as a
        boundary condition rather than being applied after the future was constructed.

        All-zero 4D rows are padding (same contract as ``centric_transform``) and
        are restored after the scale so they are not rotated as real history.
        ``[0, 0, 1, 0]`` is *not* padding: that is an ego-centric origin pose
        after ``heading_to_cos_sin``.
        """
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
            pad_mask[aug_flag] = (
                torch.sum(torch.ne(ego_past[aug_flag, :, :4], 0), dim=-1) == 0
            )

        scale_xy = scale_by_batch[aug_flag].reshape(-1, 1, 1)
        center_xy = ego_current[aug_flag, :2].unsqueeze(1)
        past_xy = ego_past[aug_flag, :, :2]
        ego_past[aug_flag, :, :2] = center_xy + (past_xy - center_xy) * scale_xy

        if pad_mask is not None:
            ego_past[pad_mask] = 0

        scale_state = scale_by_batch[aug_flag].reshape(-1, 1)
        ego_current[aug_flag, 4:8] *= scale_state

    def augment_at_tau(self, inputs, ego_future):
        """Rewrite past/current/future around a random τ; return aug tensors + flag."""
        ego_current = inputs["ego_current_state"].clone()
        ego_past = inputs["ego_agent_past"].clone()
        aug_future = ego_future.clone()
        device = ego_current.device
        dtype = ego_current.dtype
        dt = self.time_interval
        B = ego_current.shape[0]
        past_len = ego_past.shape[1]
        future_len = aug_future.shape[1]
        current_index = past_len - 1
        self.last_tau_info = None

        valid_speed = torch.abs(ego_current[:, 4]) >= 2.0
        aug_flag = (torch.rand(B, device=device) < self._augment_prob) & valid_speed

        # Apply the independent history-speed augmentation before constructing the
        # two tau bridges. Scaling about ego_current.xy is equivalent to re-centering
        # first, scaling in the centered frame, and transforming back; unlike scaling
        # about (0, 0), it also preserves the history/current join for nonzero poses.
        history_scale = torch.ones(B, device=device, dtype=dtype)
        B_aug = int(aug_flag.sum().item())
        if B_aug > 0 and self._ego_past_noise_std > 0.0:
            W = self._ego_past_noise_std
            sampled_scale = torch.normal(
                mean=1.0,
                std=W,
                size=(B_aug,),
                device=device,
            ).to(dtype=dtype)
            history_scale[aug_flag] = torch.clamp(sampled_scale, 1.0 - 2 * W, 1.0 + 2 * W)
            self._scale_history_about_current(
                ego_current=ego_current,
                ego_past=ego_past,
                aug_flag=aug_flag,
                scale_by_batch=history_scale,
            )

        for batch_index in torch.nonzero(aug_flag, as_tuple=False).flatten():
            b = int(batch_index.item())
            try:
                ok = self._augment_single_at_tau(
                    ego_current=ego_current,
                    ego_past=ego_past,
                    ego_future=aug_future,
                    wheel_base=float(inputs["ego_shape"][b, 0].item()),
                    batch_index=b,
                    current_index=current_index,
                    past_len=past_len,
                    future_len=future_len,
                    dt=dt,
                    device=device,
                    dtype=dtype,
                    history_scale=float(history_scale[b].item()),
                )
            except (RuntimeError, ValueError):
                ok = False
            if not ok:
                aug_flag[b] = False
                continue

        collision = self._check_aug_validity(ego_current, inputs)
        aug_flag = aug_flag & ~collision
        if self.last_tau_info is not None and not bool(aug_flag[0].item()):
            # Keep the sampled τ for diagnostics even when collision rejects.
            self.last_tau_info["accepted"] = False
        elif self.last_tau_info is not None:
            self.last_tau_info["accepted"] = True
        return aug_flag, ego_current, ego_past, aug_future

    def _augment_single_at_tau(
        self,
        ego_current: torch.Tensor,
        ego_past: torch.Tensor,
        ego_future: torch.Tensor,
        wheel_base: float,
        batch_index: int,
        current_index: int,
        past_len: int,
        future_len: int,
        dt: float,
        device,
        dtype,
        history_scale: float = 1.0,
    ) -> bool:
        """In-place rewrite one sample. Returns False if the window is infeasible."""
        b = batch_index
        full_xy, full_heading = self._build_full_xy_heading(
            ego_past[b], ego_current[b], ego_future[b]
        )
        n_full = full_xy.shape[0]
        times = (torch.arange(n_full, device=device, dtype=dtype) - current_index) * dt

        past_horizon_s = float((-times[0]).item())
        future_horizon_s = float(times[-1].item())
        tau_lo = max(self._tau_min_s, -past_horizon_s + dt)
        tau_hi = min(self._tau_max_s, 0.0)
        if tau_lo > tau_hi:
            return False

        # Snap τ to the discrete time grid.
        tau_s = float(torch.empty(1, device=device).uniform_(tau_lo, tau_hi).item())
        tau_idx = int(torch.argmin(torch.abs(times - tau_s)).item())
        tau_s = float(times[tau_idx].item())

        left_time = tau_s - self.refine_horizon_s
        right_time = tau_s + self.refine_horizon_s
        left_idx = int(torch.argmin(torch.abs(times - left_time)).item())
        right_idx = int(torch.argmin(torch.abs(times - right_time)).item())
        left_idx = max(0, min(left_idx, tau_idx))
        right_idx = max(tau_idx, min(right_idx, n_full - 1))
        if tau_idx - left_idx < 2 or right_idx - tau_idx < 2:
            return False

        # GT kinematics along the original polyline (yaw-rate). Speed/accel at the
        # quintic ends are signed projections onto heading, not vector magnitudes.
        # t=0 prefers the measured current state (after history scale) so it wins
        # when tau snaps to the current index.
        speed, accel, yaw_rate = self._estimate_kinematics(full_xy, full_heading, dt)
        speed[left_idx], accel[left_idx] = polyline_tangential_va(
            full_xy, full_heading, left_idx, dt
        )
        speed[right_idx], accel[right_idx] = polyline_tangential_va(
            full_xy, full_heading, right_idx, dt
        )
        speed[tau_idx], accel[tau_idx] = polyline_tangential_va(
            full_xy, full_heading, tau_idx, dt
        )
        heading0 = heading_from_cos_sin(ego_current[b, 2], ego_current[b, 3])
        speed[current_index], accel[current_index] = tangential_va(
            ego_current[b, 4:6], ego_current[b, 6:8], heading0
        )
        yaw_rate[current_index] = ego_current[b, 9]

        # Same lateral / heading / speed / accel ranges as StatePerturbation._low/_high.
        # Indices: 1=y, 2=heading, 3=vx, 5=ax. Parent leaves yaw-rate (idx 8) at 0
        # and re-derives steering from (ω, v); we mirror that at the τ mid-state.
        lateral = float(
            (
                self._low[1]
                + (self._high[1] - self._low[1]) * torch.rand((), device=device)
            ).item()
        )
        heading_off = float(
            (
                self._low[2]
                + (self._high[2] - self._low[2]) * torch.rand((), device=device)
            ).item()
        )
        speed_off = float(
            (
                self._low[3]
                + (self._high[3] - self._low[3]) * torch.rand((), device=device)
            ).item()
        )
        accel_off = float(
            (
                self._low[5]
                + (self._high[5] - self._low[5]) * torch.rand((), device=device)
            ).item()
        )
        if (
            abs(lateral) < 1e-3
            and abs(heading_off) < 1e-3
            and abs(speed_off) < 1e-3
            and abs(accel_off) < 1e-3
        ):
            return False

        # Offset pose + kinematics at τ; bridge ends keep GT so recovery is continuous.
        psi_gt = full_heading[tau_idx]
        x_gt = float(full_xy[tau_idx, 0].item())
        y_gt = float(full_xy[tau_idx, 1].item())
        psi_gt_val = float(psi_gt.item())
        x_tau = full_xy[tau_idx, 0] - lateral * torch.sin(psi_gt)
        y_tau = full_xy[tau_idx, 1] + lateral * torch.cos(psi_gt)
        psi_tau = self.normalize_angle(psi_gt + heading_off)
        psi_tau_val = float(psi_tau.item())
        v_gt = float(speed[tau_idx].item())
        a_gt = float(accel[tau_idx].item())
        w_gt = float(yaw_rate[tau_idx].item())
        v_tau = max(0.0, v_gt + speed_off)
        a_tau = a_gt + accel_off
        # Keep GT yaw-rate; steering at any state (incl. t=0) follows from (ω, v)
        # exactly as in StatePerturbation.augment.
        w_tau = w_gt

        left_states = self._pack_boundary_state(
            full_xy[left_idx], full_heading[left_idx], speed[left_idx], accel[left_idx], yaw_rate[left_idx]
        )
        mid_states = (
            x_tau,
            y_tau,
            psi_tau,
            v_tau,
            a_tau,
            w_tau,
        )
        right_states = self._pack_boundary_state(
            full_xy[right_idx],
            full_heading[right_idx],
            speed[right_idx],
            accel[right_idx],
            yaw_rate[right_idx],
        )

        # Past-side bridge [left, tau]
        n_left = tau_idx - left_idx
        T_left = n_left * dt
        times_left = torch.arange(1, n_left + 1, device=device, dtype=dtype) * dt
        traj_left = self._interpolate_segment(left_states, mid_states, T_left, times_left)
        full_xy[left_idx + 1 : tau_idx + 1] = traj_left[:, :2]
        full_heading[left_idx + 1 : tau_idx + 1] = traj_left[:, 2]

        # Future-side bridge [tau, right] — exclude tau (already set), include right.
        n_right = right_idx - tau_idx
        T_right = n_right * dt
        times_right = torch.arange(1, n_right + 1, device=device, dtype=dtype) * dt
        traj_right = self._interpolate_segment(mid_states, right_states, T_right, times_right)
        full_xy[tau_idx + 1 : right_idx + 1] = traj_right[:, :2]
        full_heading[tau_idx + 1 : right_idx + 1] = traj_right[:, 2]
        # Ensure exact mid pose (interpolation left endpoint is exclusive above).
        full_xy[tau_idx, 0] = x_tau
        full_xy[tau_idx, 1] = y_tau
        full_heading[tau_idx] = psi_tau

        # Write past (cos/sin) and future (heading).
        ego_past[b, :, 0] = full_xy[:past_len, 0]
        ego_past[b, :, 1] = full_xy[:past_len, 1]
        ego_past[b, :, 2] = torch.cos(full_heading[:past_len])
        ego_past[b, :, 3] = torch.sin(full_heading[:past_len])
        ego_future[b, :, 0] = full_xy[past_len:, 0]
        ego_future[b, :, 1] = full_xy[past_len:, 1]
        ego_future[b, :, 2] = full_heading[past_len:]

        # Re-derive current state at t=0 from the continuous path.
        ego_current[b] = self._current_state_from_full(
            full_xy,
            full_heading,
            current_index=current_index,
            dt=dt,
            wheel_base=wheel_base,
            template=ego_current[b],
        )
        # Record for visualization (batch-0 focused; last successful overwrite wins).
        self.last_tau_info = {
            "batch_index": b,
            "tau_s": tau_s,
            "tau_idx": tau_idx,
            "current_index": current_index,
            "left_idx": left_idx,
            "right_idx": right_idx,
            "left_s": float(times[left_idx].item()),
            "right_s": float(times[right_idx].item()),
            "lateral_m": lateral,
            "heading_off_rad": heading_off,
            "heading_off_deg": float(np.rad2deg(heading_off)),
            "speed_off_mps": speed_off,
            "accel_off_mps2": accel_off,
            "v_gt_mps": v_gt,
            "v_tau_mps": float(v_tau),
            "a_gt_mps2": a_gt,
            "a_tau_mps2": float(a_tau),
            "w_tau_rps": float(w_tau),
            "xy_gt": (x_gt, y_gt),
            "xy_tau": (float(x_tau.item()), float(y_tau.item())),
            "psi_gt_rad": psi_gt_val,
            "psi_tau_rad": psi_tau_val,
            "refine_horizon_s": float(self.refine_horizon_s),
            "history_scale": history_scale,
        }
        return True

    @staticmethod
    def _pack_boundary_state(xy, heading, speed, accel, yaw_rate):
        return (
            xy[0],
            xy[1],
            heading,
            speed,
            accel,
            yaw_rate,
        )

    def _build_full_xy_heading(
        self,
        ego_past_b: torch.Tensor,
        ego_current_b: torch.Tensor,
        ego_future_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate past|future; force last past row = current pose."""
        past_xy = ego_past_b[:, :2].clone()
        if ego_past_b.shape[-1] >= 4:
            past_heading = heading_from_cos_sin(ego_past_b[:, 2], ego_past_b[:, 3]).clone()
        else:
            past_heading = ego_past_b[:, 2].clone()
        past_xy[-1] = ego_current_b[:2]
        past_heading[-1] = heading_from_cos_sin(ego_current_b[2], ego_current_b[3])

        future_xy = ego_future_b[:, :2].clone()
        if ego_future_b.shape[-1] >= 4:
            future_heading = heading_from_cos_sin(ego_future_b[:, 2], ego_future_b[:, 3]).clone()
        else:
            future_heading = ego_future_b[:, 2].clone()

        full_xy = torch.cat([past_xy, future_xy], dim=0)
        full_heading = torch.cat([past_heading, future_heading], dim=0)
        return full_xy, full_heading

    def _estimate_kinematics(
        self, xy: torch.Tensor, heading: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = xy.shape[0]
        speed = torch.zeros(n, device=xy.device, dtype=xy.dtype)
        accel = torch.zeros(n, device=xy.device, dtype=xy.dtype)
        yaw_rate = torch.zeros(n, device=xy.device, dtype=xy.dtype)
        if n < 2:
            return speed, accel, yaw_rate

        dxy = xy[1:] - xy[:-1]
        seg_speed = torch.linalg.norm(dxy, dim=-1) / dt
        speed[0] = seg_speed[0]
        speed[-1] = seg_speed[-1]
        if n > 2:
            speed[1:-1] = 0.5 * (seg_speed[:-1] + seg_speed[1:])

        dheading = self.normalize_angle(heading[1:] - heading[:-1]) / dt
        yaw_rate[0] = dheading[0]
        yaw_rate[-1] = dheading[-1]
        if n > 2:
            yaw_rate[1:-1] = 0.5 * (dheading[:-1] + dheading[1:])

        dspeed = (speed[1:] - speed[:-1]) / dt
        accel[0] = dspeed[0]
        accel[-1] = dspeed[-1]
        if n > 2:
            accel[1:-1] = 0.5 * (dspeed[:-1] + dspeed[1:])
        return speed, accel, yaw_rate

    def _interpolate_segment(
        self,
        state0: tuple,
        stateT: tuple,
        duration_s: float,
        sample_times: torch.Tensor,
    ) -> torch.Tensor:
        """Quintic Hermite in time between two kinematic states → (N, 3) x,y,heading."""
        x0, y0, theta0, v0, a0, omega0 = state0
        xT, yT, thetaT, vT, aT, omegaT = stateT
        device = sample_times.device
        dtype = sample_times.dtype

        # Promote scalars to 1-batch tensors.
        def _t(v):
            if torch.is_tensor(v):
                return v.to(device=device, dtype=dtype).reshape(1)
            return torch.tensor([float(v)], device=device, dtype=dtype)

        x0, y0, theta0, v0, a0, omega0 = map(_t, (x0, y0, theta0, v0, a0, omega0))
        xT, yT, thetaT, vT, aT, omegaT = map(_t, (xT, yT, thetaT, vT, aT, omegaT))

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
        )  # (1, 6)
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

        A = _quintic_coeff_matrix(duration_s, device, dtype).unsqueeze(0)  # (1, 6, 6)
        ax = A @ sx[:, :, None]
        ay = A @ sy[:, :, None]
        powers = torch.arange(6, device=device, dtype=dtype)
        M = torch.pow(sample_times.unsqueeze(1), powers.unsqueeze(0)).unsqueeze(0)  # (1, N, 6)
        traj_x = (M @ ax)[0, :, 0]
        traj_y = (M @ ay)[0, :, 0]

        # Heading from successive chord directions (start pose → first sample → …).
        prev_x = torch.cat([x0, traj_x[:-1]], dim=0)
        prev_y = torch.cat([y0, traj_y[:-1]], dim=0)
        heading = torch.atan2(traj_y - prev_y, traj_x - prev_x)
        return torch.stack([traj_x, traj_y, heading], dim=-1)

    def _current_state_from_full(
        self,
        full_xy: torch.Tensor,
        full_heading: torch.Tensor,
        current_index: int,
        dt: float,
        wheel_base: float,
        template: torch.Tensor,
    ) -> torch.Tensor:
        out = template.clone()
        i = current_index
        out[0] = full_xy[i, 0]
        out[1] = full_xy[i, 1]
        out[2] = torch.cos(full_heading[i])
        out[3] = torch.sin(full_heading[i])

        if i == 0:
            velocity = (full_xy[1] - full_xy[0]) / dt
            acceleration = torch.zeros(2, device=full_xy.device, dtype=full_xy.dtype)
            yaw_rate = self.normalize_angle(full_heading[1] - full_heading[0]) / dt
        elif i == full_xy.shape[0] - 1:
            velocity = (full_xy[-1] - full_xy[-2]) / dt
            acceleration = torch.zeros(2, device=full_xy.device, dtype=full_xy.dtype)
            yaw_rate = self.normalize_angle(full_heading[-1] - full_heading[-2]) / dt
        else:
            velocity = (full_xy[i + 1] - full_xy[i - 1]) / (2.0 * dt)
            acceleration = (full_xy[i + 1] - 2.0 * full_xy[i] + full_xy[i - 1]) / (dt**2)
            yaw_rate = self.normalize_angle(full_heading[i + 1] - full_heading[i - 1]) / (
                2.0 * dt
            )

        speed = torch.linalg.norm(velocity)
        if speed >= 0.2:
            steering = torch.atan(yaw_rate * wheel_base / torch.abs(speed))
            steering = torch.clamp(steering, -2.0 / 3.0 * np.pi, 2.0 / 3.0 * np.pi)
        else:
            steering = torch.zeros((), device=full_xy.device, dtype=full_xy.dtype)
            yaw_rate = torch.zeros((), device=full_xy.device, dtype=full_xy.dtype)

        out[4:6] = velocity
        out[6:8] = acceleration
        out[8] = steering
        out[9] = yaw_rate
        return out
