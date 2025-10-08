#[allow(clippy::needless_pass_by_value)]
pub fn compute_normals(positions: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let count = positions.len();
    if count == 0 {
        return Vec::new();
    }

    if count == 1 {
        return vec![(0.0, 1.0)];
    }

    let mut segment_normals = Vec::with_capacity(count.saturating_sub(1));
    for window in positions.windows(2) {
        let (x0, y0) = window[0];
        let (x1, y1) = window[1];
        let dx = x1 - x0;
        let dy = y1 - y0;
        let length = (dx * dx + dy * dy).sqrt();
        if length <= f32::EPSILON {
            segment_normals.push((0.0, 1.0));
        } else {
            segment_normals.push((-dy / length, dx / length));
        }
    }

    let mut normals = Vec::with_capacity(count);
    for index in 0..count {
        let normal = if index == 0 {
            segment_normals.first().copied().unwrap_or((0.0, 1.0))
        } else if index == count - 1 {
            segment_normals
                .get(count - 2)
                .copied()
                .unwrap_or((0.0, 1.0))
        } else {
            let prev = segment_normals
                .get(index - 1)
                .copied()
                .unwrap_or((0.0, 1.0));
            let next = segment_normals.get(index).copied().unwrap_or((0.0, 1.0));
            let nx = prev.0 + next.0;
            let ny = prev.1 + next.1;
            let len = (nx * nx + ny * ny).sqrt();
            if len <= f32::EPSILON {
                next
            } else {
                (nx / len, ny / len)
            }
        };
        normals.push(normal);
    }

    normals
}
