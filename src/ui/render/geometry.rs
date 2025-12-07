//! Geometry utilities for rendering.

use crate::ui::render::common::{ClipTransform, SdfVertex};

const MIN_SEGMENT_LENGTH: f32 = 0.1;
const MAX_MITER_RATIO: f32 = 4.0;

/// Default feather distance for antialiased lines.
pub const DEFAULT_FEATHER: f32 = 1.0;

/// Joins triangle strips with degenerate triangles for batched draw calls.
pub fn append_strip(dest: &mut Vec<SdfVertex>, strip: Vec<SdfVertex>) {
    if strip.is_empty() {
        return;
    }
    if !dest.is_empty()
        && let Some(last) = dest.last().copied()
    {
        let mut iter = strip.into_iter();
        let first = iter.next().unwrap();
        dest.push(last);
        dest.push(last);
        dest.push(first);
        dest.push(first);
        dest.extend(iter);
        return;
    }
    dest.extend(strip);
}

pub fn compute_normals(positions: &[(f32, f32)]) -> Vec<(f32, f32)> {
    match positions.len() {
        0 => Vec::new(),
        1 => vec![(0.0, 1.0)],
        _ => {
            let segment_normals: Vec<_> = positions
                .windows(2)
                .map(|w| {
                    let (dx, dy) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
                    let len_sq = dx * dx + dy * dy;
                    if len_sq < MIN_SEGMENT_LENGTH * MIN_SEGMENT_LENGTH {
                        None
                    } else {
                        let inv_len = len_sq.sqrt().recip();
                        Some((-dy * inv_len, dx * inv_len))
                    }
                })
                .collect();

            (0..positions.len())
                .map(|i| {
                    let prev = if i > 0 { segment_normals[i - 1] } else { None };
                    let next = segment_normals.get(i).copied().flatten();

                    match (prev, next) {
                        (Some((px, py)), Some((nx, ny))) => {
                            let (sx, sy) = (px + nx, py + ny);
                            let len_sq = sx * sx + sy * sy;
                            if len_sq <= f32::EPSILON * f32::EPSILON {
                                (nx, ny)
                            } else {
                                let inv_len = len_sq.sqrt().recip();
                                let miter_ratio = inv_len * 2.0;
                                if miter_ratio > MAX_MITER_RATIO {
                                    (nx, ny)
                                } else {
                                    (sx * inv_len, sy * inv_len)
                                }
                            }
                        }
                        (Some(n), None) | (None, Some(n)) => n,
                        (None, None) => {
                            if let Some(n) = segment_normals[i..].iter().find_map(|&x| x) {
                                return n;
                            }
                            if let Some(n) = segment_normals[..i].iter().rev().find_map(|&x| x) {
                                return n;
                            }
                            (0.0, 1.0)
                        }
                    }
                })
                .collect()
        }
    }
}

/// Builds an antialiased polyline for `TriangleStrip` topology.
pub fn build_aa_line_strip(
    positions: &[(f32, f32)],
    stroke_width: f32,
    feather: f32,
    color: [f32; 4],
    clip: &ClipTransform,
) -> Vec<SdfVertex> {
    if positions.len() < 2 {
        return Vec::new();
    }

    let normals = compute_normals(positions);
    let half = stroke_width.max(0.1) * 0.5;
    let outer = half + feather;

    let mut vertices = Vec::with_capacity(positions.len() * 2);

    for ((x, y), (nx, ny)) in positions.iter().zip(normals.iter()) {
        let offset_x = nx * outer;
        let offset_y = ny * outer;

        vertices.push(SdfVertex::antialiased(
            clip.to_clip(x - offset_x, y - offset_y),
            color,
            -outer,
            half,
            feather,
        ));
        vertices.push(SdfVertex::antialiased(
            clip.to_clip(x + offset_x, y + offset_y),
            color,
            outer,
            half,
            feather,
        ));
    }

    vertices
}

/// Like `build_aa_line_strip` but with per-vertex colors.
pub fn build_aa_line_strip_colored(
    positions: &[(f32, f32)],
    colors: &[[f32; 4]],
    stroke_width: f32,
    feather: f32,
    clip: &ClipTransform,
) -> Vec<SdfVertex> {
    if positions.len() < 2 || colors.len() < positions.len() {
        return Vec::new();
    }

    let normals = compute_normals(positions);
    let half = stroke_width.max(0.1) * 0.5;
    let outer = half + feather;

    let mut vertices = Vec::with_capacity(positions.len() * 2);

    for (((x, y), (nx, ny)), color) in positions.iter().zip(normals.iter()).zip(colors.iter()) {
        let offset_x = nx * outer;
        let offset_y = ny * outer;

        vertices.push(SdfVertex::antialiased(
            clip.to_clip(x - offset_x, y - offset_y),
            *color,
            -outer,
            half,
            feather,
        ));
        vertices.push(SdfVertex::antialiased(
            clip.to_clip(x + offset_x, y + offset_y),
            *color,
            outer,
            half,
            feather,
        ));
    }

    vertices
}

/// Builds an antialiased polyline for `TriangleList` topology.
pub fn build_aa_line_list(
    positions: &[(f32, f32)],
    stroke_width: f32,
    feather: f32,
    color: [f32; 4],
    clip: &ClipTransform,
) -> Vec<SdfVertex> {
    if positions.len() < 2 {
        return Vec::new();
    }

    let normals = compute_normals(positions);
    let half = stroke_width.max(0.1) * 0.5;
    let outer = half + feather;

    // Each segment emits 6 vertices (2 triangles)
    let mut vertices = Vec::with_capacity((positions.len() - 1) * 6);
    let mut prev: Option<(SdfVertex, SdfVertex)> = None;

    for ((x, y), (nx, ny)) in positions.iter().zip(normals.iter()) {
        let offset_x = nx * outer;
        let offset_y = ny * outer;

        let left = SdfVertex::antialiased(
            clip.to_clip(x - offset_x, y - offset_y),
            color,
            -outer,
            half,
            feather,
        );
        let right = SdfVertex::antialiased(
            clip.to_clip(x + offset_x, y + offset_y),
            color,
            outer,
            half,
            feather,
        );

        if let Some((l0, r0)) = prev {
            vertices.extend_from_slice(&[l0, r0, right, l0, right, left]);
        }
        prev = Some((left, right));
    }

    vertices
}
