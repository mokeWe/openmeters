use std::fmt;

#[derive(Clone)]
pub struct RingBuffer<T> {
    slots: Vec<Option<T>>,
    head: usize,
    len: usize,
}

impl<T> RingBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "RingBuffer capacity must be greater than zero"
        );
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || None);

        Self {
            slots,
            head: 0,
            len: 0,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity()
    }

    pub fn push(&mut self, value: T) -> Option<T> {
        let capacity = self.capacity();
        debug_assert!(
            capacity > 0,
            "RingBuffer capacity must remain greater than zero"
        );

        let idx = if self.is_full() {
            let current = self.head;
            self.head = (self.head + 1) % capacity;
            current
        } else {
            let idx = (self.head + self.len) % capacity;
            debug_assert!(
                self.slots[idx].is_none(),
                "Slot should be vacant when buffer is not full"
            );
            self.len += 1;
            idx
        };

        self.slots[idx].replace(value)
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let idx = self.head;
        let value = self.slots[idx].take();
        self.head = (self.head + 1) % self.capacity();
        self.len -= 1;
        value
    }

    pub fn peek(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            self.slots[self.head].as_ref()
        }
    }

    pub fn clear(&mut self) {
        if self.is_empty() {
            return;
        }

        for slot in &mut self.slots {
            *slot = None;
        }
        self.head = 0;
        self.len = 0;
    }

    pub fn push_iter<I>(&mut self, iter: I) -> usize
    where
        I: IntoIterator<Item = T>,
    {
        let mut overwritten = 0;
        for value in iter {
            if self.push(value).is_some() {
                overwritten += 1;
            }
        }
        overwritten
    }

    pub fn iter(&self) -> RingBufferIter<'_, T> {
        RingBufferIter {
            buffer: self,
            offset: 0,
            remaining: self.len,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for RingBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let items: Vec<&T> = self.iter().collect();
        f.debug_struct("RingBuffer")
            .field("capacity", &self.capacity())
            .field("len", &self.len())
            .field("items", &items)
            .finish()
    }
}

/// Immutable iterator over the contents of a `RingBuffer`.
pub struct RingBufferIter<'a, T> {
    buffer: &'a RingBuffer<T>,
    offset: usize,
    remaining: usize,
}

impl<'a, T> Iterator for RingBufferIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let idx = (self.buffer.head + self.offset) % self.buffer.capacity();
        self.offset += 1;
        self.remaining -= 1;
        // SAFETY: slots within the active range are always populated.
        self.buffer.slots[idx].as_ref()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> ExactSizeIterator for RingBufferIter<'a, T> {}

impl<'a, T> IntoIterator for &'a RingBuffer<T> {
    type Item = &'a T;
    type IntoIter = RingBufferIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Owning iterator that drains the buffer in FIFO order.
pub struct IntoIter<T> {
    buffer: RingBuffer<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.buffer.len(), Some(self.buffer.len()))
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> IntoIterator for RingBuffer<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { buffer: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_with_capacity_and_is_empty() {
        let buffer: RingBuffer<i32> = RingBuffer::with_capacity(4);
        assert_eq!(buffer.capacity(), 4);
        assert!(buffer.is_empty());
    }

    #[test]
    #[should_panic(expected = "RingBuffer capacity must be greater than zero")]
    fn zero_capacity_panics() {
        let _buffer: RingBuffer<i32> = RingBuffer::with_capacity(0);
    }

    #[test]
    fn push_and_pop_in_fifo_order() {
        let mut buffer = RingBuffer::with_capacity(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
        assert!(buffer.is_empty());
    }

    #[test]
    fn push_overwrites_when_full() {
        let mut buffer = RingBuffer::with_capacity(2);
        assert!(buffer.push(1).is_none());
        assert!(buffer.push(2).is_none());
        assert_eq!(buffer.push(3), Some(1));
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
    }

    #[test]
    fn peek_returns_oldest_without_removing() {
        let mut buffer = RingBuffer::with_capacity(3);
        buffer.push(10);
        buffer.push(20);
        assert_eq!(buffer.peek(), Some(&10));
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.pop(), Some(10));
        assert_eq!(buffer.peek(), Some(&20));
    }

    #[test]
    fn clear_removes_all_items() {
        let mut buffer = RingBuffer::with_capacity(3);
        buffer.push_iter([1, 2, 3]);
        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.pop(), None);
        assert!(buffer.push(4).is_none());
        assert_eq!(buffer.pop(), Some(4));
    }

    #[test]
    fn iterator_reflects_current_state() {
        let mut buffer = RingBuffer::with_capacity(4);
        buffer.push_iter(1..=4);
        let collected: Vec<_> = buffer.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3, 4]);

        buffer.push_iter([5, 6]);
        let collected: Vec<_> = buffer.iter().copied().collect();
        assert_eq!(collected, vec![3, 4, 5, 6]);
    }

    #[test]
    fn into_iter_drains_in_order() {
        let mut buffer = RingBuffer::with_capacity(3);
        buffer.push_iter(['a', 'b', 'c']);
        let collected: String = buffer.into_iter().collect();
        assert_eq!(collected, "abc");
    }
}
