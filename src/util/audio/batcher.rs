const MAX_RECYCLED_BUFFERS: usize = 4;

#[derive(Default)]
pub struct SampleBatcher {
    target_samples: usize,
    total_samples: usize,
    chunks: Vec<Vec<f32>>,
    recycle: Vec<Vec<f32>>,
}

impl SampleBatcher {
    pub fn new(target_samples: usize) -> Self {
        Self {
            target_samples,
            ..Self::default()
        }
    }

    pub fn push(&mut self, chunk: Vec<f32>) {
        self.total_samples = self.total_samples.saturating_add(chunk.len());
        self.chunks.push(chunk);
    }

    pub fn should_flush(&self) -> bool {
        self.total_samples >= self.target_samples
    }

    pub fn clear(&mut self) {
        let drained: Vec<_> = self.chunks.drain(..).collect();
        for chunk in drained {
            Self::stash_recycle(&mut self.recycle, chunk);
        }
        self.total_samples = 0;
    }

    pub fn take(&mut self) -> Option<Vec<f32>> {
        if self.total_samples == 0 {
            return None;
        }

        self.total_samples = 0;

        if self.chunks.len() == 1 {
            return self.chunks.pop();
        }

        let total_samples = self.chunks.iter().map(|c| c.len()).sum();
        let mut batch = self.reuse_buffer(total_samples);

        for chunk in self.chunks.drain(..) {
            batch.extend_from_slice(&chunk);
            Self::stash_recycle(&mut self.recycle, chunk);
        }

        Some(batch)
    }

    fn reuse_buffer(&mut self, needed: usize) -> Vec<f32> {
        if let Some(mut recycled) = self.recycle.pop() {
            recycled.clear();
            if recycled.capacity() < needed {
                recycled.reserve(needed - recycled.capacity());
            }
            recycled
        } else {
            Vec::with_capacity(needed)
        }
    }

    fn stash_recycle(recycle: &mut Vec<Vec<f32>>, mut chunk: Vec<f32>) {
        if recycle.len() < MAX_RECYCLED_BUFFERS {
            chunk.clear();
            recycle.push(chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SampleBatcher;

    #[test]
    fn batches_and_reuses_buffers() {
        let mut batcher = SampleBatcher::new(4);
        batcher.push(vec![0.0, 1.0]);
        assert!(!batcher.should_flush());
        batcher.push(vec![2.0, 3.0]);
        assert!(batcher.should_flush());

        let batch = batcher.take().expect("batch should be available");
        assert_eq!(batch, vec![0.0, 1.0, 2.0, 3.0]);
        assert!(batcher.take().is_none());

        batcher.push(vec![4.0, 5.0]);
        batcher.push(vec![6.0, 7.0]);
        let second = batcher.take().expect("second batch available");
        assert_eq!(second, vec![4.0, 5.0, 6.0, 7.0]);
    }
}
