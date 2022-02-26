import os
import tqdm
import numpy as np
from PIL import Image

from model import *

# data
cp_name = "test"
ds_path = "datasets/pkmn_small_ds.npy"

num_epochs = 10001
max_episode_length = 100
discount = .9
batch_size = 1


# cache
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def main():
    tf.config.list_physical_devices('GPU')

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # load data
    data = np.load(ds_path)
    data = data.astype("float32")
    data /= 255

    train_dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0]).batch(batch_size)

    # setup model
    disc = Discriminator()
    rlgen = RLGenerator()

    train_loop(rlgen, disc, data[0].shape, train_dataset)


def train_step(rlgen, disc, input_shape, images):
    with tf.GradientTape() as tape:
        shape = [batch_size]
        for s in input_shape:
            shape.append(s)
        start = tf.random.uniform(shape, 0, 1)
        p, v, r, s = run_episode(start, rlgen, disc)
        returns = get_expected_return(r)
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [p, v, returns]]
        loss = ac_loss(action_probs, values, returns)

        loss += discriminator_loss(disc.call(images), disc.call(s))

        # Compute the gradients from the loss
    grads = tape.gradient(loss, rlgen.trainable_variables)

    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, rlgen.trainable_variables))
    return s


def train_loop(rlgen, disc, input_shape, dataset):
    print("Begin training loop")
    with tqdm.trange(num_epochs) as t:
        for e in t:
            i = 0
            for b in dataset:
                print(i)
                s = train_step(rlgen, disc, input_shape, b)
                i += 1

            # save state
            img = s[0] * 255
            img = img.numpy().astype("uint8")

            img = Image.fromarray(img, mode="RGBA")
            img.save(os.getcwd() + "/" + "Test Img" + str(e) + ".png", "png")
        t.set_description(f'Epoch {e}')


def run_episode(start, agent, discriminator):
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    state = start

    for t in range(max_episode_length):
        a, p, state, c = agent.call(state)
        state = tf.identity(state)
        values = values.write(t, tf.squeeze(c))
        action_probs = action_probs.write(t, tf.reduce_sum(p))

        r = tf.stop_gradient(discriminator.call(state))

        rewards = rewards.write(t, r[0, 0])
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    return action_probs, values, rewards, state


def get_expected_return(
        rewards: tf.Tensor,
        standardize: bool = True) -> tf.Tensor:

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + discount * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + 1e-9))

    return returns


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def ac_loss(action_log_probs, predicted_value, rewards):
    advantage = rewards - predicted_value
    a_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    c_loss = huber_loss(predicted_value, rewards)

    return a_loss + c_loss;


if __name__ == "__main__":
    main()
