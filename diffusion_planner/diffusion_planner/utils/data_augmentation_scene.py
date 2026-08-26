"""Shared ego-centric scene transforms and t=0 collision checks for augmentation."""

from __future__ import annotations

import torch

from diffusion_planner.utils.data_augmentation_geom import (
    _rect_corners,
    _sat_signed_distance,
    _segments_intersect_rect,
    heading_transform,
    vector_transform,
)
from diffusion_planner.utils.unicycle_accel_curvature import smoothing_future_trajectory


def get_transform_matrix_batch(cur_state: torch.Tensor) -> torch.Tensor:
    """Rotation matrix R(-heading) from ego ``(cos, sin)`` rows. Shape ``(B, 2, 2)``."""
    processed_input = torch.column_stack((cur_state[:, 2], cur_state[:, 3]))
    reshaping_tensor = torch.tensor(
        [[1, 0, 0, 1], [0, 1, -1, 0]],
        dtype=torch.float32,
        device=processed_input.device,
    )
    return (processed_input @ reshaping_tensor).reshape(-1, 2, 2)


def check_aug_validity(aug_ego_state: torch.Tensor, inputs: dict) -> torch.Tensor:
    """
    Returns ``[B]`` bool — True where the augmented ego position is invalid.

    Invalid conditions:
      1. Ego polygon overlaps with a neighbour agent polygon.
      2. Ego polygon intersects a road-border segment from ``line_strings``.
    """
    B = aug_ego_state.shape[0]
    device = aug_ego_state.device
    dtype = aug_ego_state.dtype

    ego_shape = inputs["ego_shape"].to(device=device, dtype=dtype)
    ego_length = ego_shape[:, 1:2]
    ego_width = ego_shape[:, 2:3]

    ego_rect = torch.cat([aug_ego_state[:, :4], ego_length, ego_width], dim=-1)
    ego_corners = _rect_corners(ego_rect)

    collision = torch.zeros(B, dtype=torch.bool, device=device)

    if "neighbor_agents_past" in inputs:
        nbr = inputs["neighbor_agents_past"][:, :, -1, :]
        N = nbr.shape[1]
        valid = torch.sum(torch.ne(nbr[:, :, :4], 0), dim=-1) > 0
        if valid.any():
            nbr_rect = torch.cat(
                [nbr[:, :, :4], nbr[:, :, 7:8], nbr[:, :, 6:7]],
                dim=-1,
            )
            dists = _sat_signed_distance(
                _rect_corners(ego_rect.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 6)),
                _rect_corners(nbr_rect.reshape(B * N, 6)),
            ).reshape(B, N)
            collision = collision | ((dists < 0) & valid).any(dim=1)

    if "line_strings" in inputs:
        line_strings = inputs["line_strings"]
        if line_strings.shape[-1] >= 4:
            pts = line_strings[..., :2]
            is_road_border = (line_strings[..., 3] > 0.5).any(dim=-1)
            point_valid = torch.norm(pts, dim=-1) > 1e-6
            seg_valid = (
                point_valid[:, :, :-1] & point_valid[:, :, 1:] & is_road_border[:, :, None]
            )
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


def centric_transform(
    inputs: dict,
    ego_future: torch.Tensor,
    neighbors_future: torch.Tensor,
    *,
    use_smoothing_future_trajectory: bool,
    transform_ego_past: bool = False,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Rewrite the scene into the augmented ego frame at ``ego_current_state``."""
    cur_state = inputs["ego_current_state"].clone()
    center_xy = cur_state[:, :2]
    transform_matrix = get_transform_matrix_batch(cur_state)

    inputs["ego_current_state"][..., :2] = vector_transform(
        inputs["ego_current_state"][..., :2], transform_matrix, center_xy
    )
    inputs["ego_current_state"][..., 2:4] = vector_transform(
        inputs["ego_current_state"][..., 2:4], transform_matrix
    )
    inputs["ego_current_state"][..., 4:6] = vector_transform(
        inputs["ego_current_state"][..., 4:6], transform_matrix
    )
    inputs["ego_current_state"][..., 6:8] = vector_transform(
        inputs["ego_current_state"][..., 6:8], transform_matrix
    )

    ego_future[..., :2] = vector_transform(ego_future[..., :2], transform_matrix, center_xy)
    ego_future[..., 2] = heading_transform(ego_future[..., 2], transform_matrix)

    if transform_ego_past:
        ego_past_mask = torch.sum(torch.ne(inputs["ego_agent_past"][..., :4], 0), dim=-1) == 0
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
            ego_future[..., :2],
            torch.cos(ego_future[..., 2:3]),
            torch.sin(ego_future[..., 2:3]),
        ],
        dim=-1,
    )

    if use_smoothing_future_trajectory:
        ego_future4d = smoothing_future_trajectory(
            ego_past4d, inputs["ego_current_state"], ego_future4d
        )

    ego_future = torch.cat(
        [
            ego_future4d[..., :2],
            torch.atan2(ego_future4d[..., 3], ego_future4d[..., 2]).unsqueeze(-1),
        ],
        dim=-1,
    )
    inputs["ego_agent_future"] = ego_future

    mask = torch.sum(torch.ne(inputs["goal_pose"][..., :2], 0), dim=-1) == 0
    inputs["goal_pose"][..., :2] = vector_transform(
        inputs["goal_pose"][..., :2], transform_matrix, center_xy
    )
    inputs["goal_pose"][..., 2:4] = vector_transform(
        inputs["goal_pose"][..., 2:4], transform_matrix
    )
    inputs["goal_pose"][mask] = 0.0

    mask = torch.sum(torch.ne(inputs["neighbor_agents_past"][..., :6], 0), dim=-1) == 0
    inputs["neighbor_agents_past"][..., :2] = vector_transform(
        inputs["neighbor_agents_past"][..., :2], transform_matrix, center_xy
    )
    inputs["neighbor_agents_past"][..., 2:4] = vector_transform(
        inputs["neighbor_agents_past"][..., 2:4], transform_matrix
    )
    inputs["neighbor_agents_past"][..., 4:6] = vector_transform(
        inputs["neighbor_agents_past"][..., 4:6], transform_matrix
    )
    inputs["neighbor_agents_past"][mask] = 0.0

    mask = torch.sum(torch.ne(neighbors_future[..., :2], 0), dim=-1) == 0
    neighbors_future[..., :2] = vector_transform(
        neighbors_future[..., :2], transform_matrix, center_xy
    )
    if neighbors_future.shape[-1] == 4:
        neighbors_future[..., 2:4] = vector_transform(
            neighbors_future[..., 2:4], transform_matrix
        )
    else:
        neighbors_future[..., 2] = heading_transform(neighbors_future[..., 2], transform_matrix)
    neighbors_future[mask] = 0.0

    mask = torch.sum(torch.ne(inputs["lanes"][..., :8], 0), dim=-1) == 0
    inputs["lanes"][..., :2] = vector_transform(inputs["lanes"][..., :2], transform_matrix, center_xy)
    inputs["lanes"][..., 2:4] = vector_transform(inputs["lanes"][..., 2:4], transform_matrix)
    inputs["lanes"][..., 4:6] = vector_transform(inputs["lanes"][..., 4:6], transform_matrix)
    inputs["lanes"][..., 6:8] = vector_transform(inputs["lanes"][..., 6:8], transform_matrix)
    inputs["lanes"][mask] = 0.0

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

    mask = torch.sum(torch.ne(inputs["polygons"], 0), dim=-1) == 0
    inputs["polygons"][..., :2] = vector_transform(
        inputs["polygons"][..., :2], transform_matrix, center_xy
    )
    inputs["polygons"][mask] = 0.0

    mask = torch.sum(torch.ne(inputs["line_strings"], 0), dim=-1) == 0
    inputs["line_strings"][..., :2] = vector_transform(
        inputs["line_strings"][..., :2], transform_matrix, center_xy
    )
    inputs["line_strings"][mask] = 0.0

    mask = torch.sum(torch.ne(inputs["static_objects"][..., :10], 0), dim=-1) == 0
    inputs["static_objects"][..., :2] = vector_transform(
        inputs["static_objects"][..., :2], transform_matrix, center_xy
    )
    inputs["static_objects"][..., 2:4] = vector_transform(
        inputs["static_objects"][..., 2:4], transform_matrix
    )
    inputs["static_objects"][mask] = 0.0

    return inputs, ego_future, neighbors_future
