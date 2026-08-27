"""Shared geometry helpers for data augmentation."""

import torch

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
