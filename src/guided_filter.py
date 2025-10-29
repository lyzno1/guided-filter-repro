from __future__ import annotations

import cv2
import numpy as np


def _box_filter(src: np.ndarray, radius: int) -> np.ndarray:
    """Fast box filter using OpenCV with reflect padding."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    return cv2.boxFilter(src, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)


def guided_filter(I: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Guided filter implementation following He et al. 2012.

    Args:
        I: Guidance image (H×W or H×W×3) in float32 [0, 1].
        p: Filtering input (same spatial size as I, 1 or 3 channels).
        radius: Window radius r.
        eps: Regularization parameter ε.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if eps < 0:
        raise ValueError("eps must be non-negative.")

    I = np.asarray(I, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)

    if I.ndim == 2:
        guidance = I
        return _guided_filter_gray(guidance, p, radius, eps)

    if I.ndim == 3 and I.shape[2] == 3:
        return _guided_filter_color(I, p, radius, eps)

    raise ValueError("Guidance image must have shape (H, W) or (H, W, 3).")


def _guided_filter_gray(I: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided filter for a grayscale guidance image."""
    if p.ndim == 3 and p.shape[2] not in (1, 3):
        raise ValueError("Input image p must have 1 or 3 channels.")
    if p.shape[:2] != I.shape[:2]:
        raise ValueError("I and p must have identical spatial dimensions.")

    mean_I = _box_filter(I, radius)
    mean_p = _box_filter(p, radius)
    mean_Ip = _box_filter(I * p, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    var_I = _box_filter(I * I, radius) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    q = mean_a * I[..., None] if p.ndim == 3 else mean_a * I
    q = q + mean_b

    if p.ndim == 3 and p.shape[2] == 1:
        return q.reshape(I.shape[0], I.shape[1], 1)
    return q


def _guided_filter_color(I: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided filter with RGB guidance, solving Σ a = cov."""
    if p.ndim == 2:
        p = p[..., None]
    if p.ndim != 3 or p.shape[2] not in (1, 3):
        raise ValueError("Input image p must have shape (H, W) or (H, W, C) with C=1 or 3.")
    if p.shape[:2] != I.shape[:2]:
        raise ValueError("I and p must share height and width.")

    eps_eye = eps * np.identity(3, dtype=np.float32)

    mean_I = np.dstack([_box_filter(I[:, :, c], radius) for c in range(3)])
    mean_p = _box_filter(p, radius)

    cov_I = np.empty(I.shape[:2] + (3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(i, 3):
            cov = _box_filter(I[:, :, i] * I[:, :, j], radius) - mean_I[:, :, i] * mean_I[:, :, j]
            cov_I[:, :, i, j] = cov
            cov_I[:, :, j, i] = cov
    channels_p = p.shape[2]
    cov_Ip = np.empty(I.shape[:2] + (3, channels_p), dtype=np.float32)
    for c in range(channels_p):
        for k in range(3):
            cov_Ip[:, :, k, c] = _box_filter(I[:, :, k] * p[:, :, c], radius) - mean_I[:, :, k] * mean_p[:, :, c]

    h, w = I.shape[:2]
    cov_I = cov_I.reshape(-1, 3, 3) + eps_eye
    cov_Ip = cov_Ip.reshape(-1, 3, channels_p)

    inv_cov_I = np.linalg.inv(cov_I)
    a = np.einsum("nij,njk->nik", inv_cov_I, cov_Ip)
    a = a.reshape(h, w, 3, channels_p)

    mean_I_expanded = mean_I[:, :, :, None]
    b = mean_p - np.sum(a * mean_I_expanded, axis=2)

    mean_a = np.empty_like(a)
    for c in range(channels_p):
        for k in range(3):
            mean_a[:, :, k, c] = _box_filter(a[:, :, k, c], radius)
    mean_b = np.dstack([_box_filter(b[:, :, c], radius) for c in range(channels_p)])

    q = np.sum(mean_a * I[:, :, :, None], axis=2) + mean_b

    if q.shape[2] == 1:
        return q.reshape(h, w)
    return q
