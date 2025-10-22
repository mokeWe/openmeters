use async_channel::Receiver as AsyncReceiver;
use iced::Subscription;
use iced::advanced::subscription::{EventStream, Hasher, Recipe, from_recipe};
use iced::futures::{self, StreamExt};
use std::fmt;
use std::hash::Hasher as _;
use std::sync::Arc;

/// Build an `iced` subscription that forwards every value produced by the given async channel.
pub fn channel_subscription<T>(receiver: Arc<AsyncReceiver<T>>) -> Subscription<T>
where
    T: Send + 'static,
{
    from_recipe(ChannelRecipe { receiver })
}

#[derive(Clone)]
struct ChannelRecipe<T> {
    receiver: Arc<AsyncReceiver<T>>,
}

impl<T> Recipe for ChannelRecipe<T>
where
    T: Send + 'static,
{
    type Output = T;

    fn hash(&self, state: &mut Hasher) {
        let ptr = Arc::as_ptr(&self.receiver) as usize;
        state.write(&ptr.to_ne_bytes());
    }

    fn stream(self: Box<Self>, _input: EventStream) -> futures::stream::BoxStream<'static, T> {
        let receiver = Arc::clone(&self.receiver);
        futures::stream::unfold(receiver, |receiver| async move {
            match receiver.recv().await {
                Ok(value) => Some((value, receiver)),
                Err(_) => None,
            }
        })
        .boxed()
    }
}

impl<T> fmt::Debug for ChannelRecipe<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChannelRecipe").finish_non_exhaustive()
    }
}
