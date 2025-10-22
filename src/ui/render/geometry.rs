//! geometry utilities for rendering.

pub fn compute_normals(positions: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let count = positions.len();
    if count == 0 {
        return Vec::new();
    }

    if count == 1 {
        return vec![(0.0, 1.0)];
    }

    let mut normals = Vec::with_capacity(count);
    let segment_normals: Vec<_> = positions
        .windows(2)
        .map(|window| segment_normal(window[0], window[1]))
        .collect();

    let mut prev_normal: Option<(f32, f32)> = None;
    let mut next_idx = 0;

    for index in 0..count {
        while next_idx < segment_normals.len() && segment_normals[next_idx].is_none() {
            next_idx += 1;
        }

        let next_normal = if next_idx < segment_normals.len() {
            segment_normals[next_idx]
        } else {
            None
        };

        let normal = match (prev_normal, next_normal) {
            (Some(prev), Some(next)) => {
                let nx = prev.0 + next.0;
                let ny = prev.1 + next.1;
                let len = (nx * nx + ny * ny).sqrt();
                if len <= f32::EPSILON {
                    next
                } else {
                    let inv_len = len.recip();
                    (nx * inv_len, ny * inv_len)
                }
            }
            (Some(prev), None) => prev,
            (None, Some(next)) => next,
            (None, None) => (0.0, 1.0),
        };

        normals.push(normal);

        if index < segment_normals.len()
            && let Some(current) = segment_normals[index] {
                prev_normal = Some(current);
            }

        if next_idx == index {
            next_idx += 1;
        }
    }

    normals
}

fn segment_normal(a: (f32, f32), b: (f32, f32)) -> Option<(f32, f32)> {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let length = (dx * dx + dy * dy).sqrt();
    if length <= f32::EPSILON {
        None
    } else {
        Some((-dy / length, dx / length))
    }
}
