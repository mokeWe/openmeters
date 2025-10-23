//! geometry utilities for rendering.

pub fn compute_normals(positions: &[(f32, f32)]) -> Vec<(f32, f32)> {
    match positions.len() {
        0 => Vec::new(),
        1 => vec![(0.0, 1.0)],
        _ => {
            // Compute segment normals inline
            let segment_normals: Vec<_> = positions
                .windows(2)
                .map(|w| {
                    let (dx, dy) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
                    let len_sq = dx * dx + dy * dy;
                    if len_sq <= f32::EPSILON * f32::EPSILON {
                        None
                    } else {
                        let inv_len = len_sq.sqrt().recip();
                        Some((-dy * inv_len, dx * inv_len))
                    }
                })
                .collect();

            // Average adjacent segment normals to get vertex normals
            (0..positions.len())
                .map(|i| {
                    let prev = if i > 0 { segment_normals[i - 1] } else { None };
                    let next = segment_normals.get(i).copied().flatten();

                    match (prev, next) {
                        (Some((px, py)), Some((nx, ny))) => {
                            let (sx, sy) = (px + nx, py + ny);
                            let len_sq = sx * sx + sy * sy;
                            if len_sq <= f32::EPSILON * f32::EPSILON {
                                (nx, ny) // Fall back to next if sum is degenerate
                            } else {
                                let inv_len = len_sq.sqrt().recip();
                                (sx * inv_len, sy * inv_len)
                            }
                        }
                        (Some(n), None) | (None, Some(n)) => n,
                        (None, None) => (0.0, 1.0),
                    }
                })
                .collect()
        }
    }
}
