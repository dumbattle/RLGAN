import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

import settings
from DenseNet import dense_block
import numpy as np
import cv2
from utils import imshow


class DiscreteAgent(tf.keras.Model):
    def __init__(self, input_shape):
        super(DiscreteAgent, self).__init__()
        # actor
        a = layers.Input(shape=input_shape)
        x = dense_block(a, 0)[0]
        x = layers.Conv2D(4 * (settings.DiscreteAgent.max_action * 2 + 1), 1, activation="tanh", padding="same")(x)

        self.actor = tf.keras.Model(inputs=a, outputs=x)

        # critic
        self.critic = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(1)
        ])

    def call(self, x):
        actor = self.actor(x)
        actor = tf.reshape(actor, [*actor.shape[0:-1], 4, -1])
        actor = tf.nn.softmax(actor)
        critic = self.critic(x)

        return actor, critic

    @staticmethod
    def update_img(img, action):
        action = tf.cast(action, tf.float32)
        action -= settings.DiscreteAgent.max_action
        action /= 255.

        return img + action

    @staticmethod
    def sample(dist):
        # needs to be 2D
        rs_dist = tf.reshape(dist, [-1, dist.shape[-1]])
        sample = tf.random.categorical(rs_dist, 1, dtype=tf.int32)

        # reshape back to original shape
        return tf.reshape(sample, dist.shape[0:-1])

    @staticmethod
    def get_mask(x):
        return tf.where(tf.random.uniform(x.shape, 0, 1) < .2, 0, 1)

    @staticmethod
    def log_prob(dist, action):
        action = tf.cast(action, dtype=tf.int32)
        dims = [tf.range(x) for x in dist.shape[:-1]]
        ind = tf.stack(tf.meshgrid(*dims, indexing='ij') + [action], axis=-1)
        probs = tf.gather_nd(dist, ind)
        return tf.math.log(probs)


class _Experience:
    def __init__(self, state, action, reward, mask):
        self.state = state
        self.action = action
        self.reward = reward
        self.mask = mask


class _Trajectory:
    def __init__(self):
        self.experiences = []

    def add_experience(self, exp):
        self.experiences.append(exp)

    def end(self):
        gamma = .9
        reward = self.experiences[-1].reward
        rs = []
        for i in reversed(range(len(self.experiences))):
            reward = reward * gamma + self.experiences[i].reward * (1 - gamma)
            self.experiences[i].reward = reward
            rs.append(reward)


class _Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size):
        # Buffer initialization
        self.state_buffer = np.zeros(
            (size, *observation_dimensions), dtype=np.float32
        )

        self.action_buffer = np.zeros((size, *observation_dimensions), dtype=np.float32)
        self.mask_buffer = np.zeros((size, *observation_dimensions), dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)

        self.pointer = 0

    def add_trajectory(self, traj):
        for exp in traj.experiences:
            self.state_buffer[self.pointer] = exp.state
            self.action_buffer[self.pointer] = exp.action
            self.reward_buffer[self.pointer] = exp.reward
            self.mask_buffer[self.pointer] = exp.mask

    def reset(self):
        self.pointer = 0


class DiscreteTrainer:
    def __init__(self, rlgen, disc, input_shape, fake_buffer):
        self.fake_buffer = fake_buffer
        self.buffer = _Buffer(input_shape, settings.DiscreteAgent.Training.max_episode_length)
        self.agent: DiscreteAgent = rlgen
        self.input_shape = input_shape
        self.disc = disc
        self.opt = tf.keras.optimizers.Adam(learning_rate=.00001)

    def _run_episode(self, episode_num):
        state = tf.random.uniform([1, *self.input_shape], 0, 1)
        traj = _Trajectory()
        reward = None
        final_reward = None
        pbar = tqdm(range(settings.RLGen.Training.max_episode_length), f"Episode {episode_num}")
        for _ in pbar:
            dist, _ = self.agent.call(state)
            action = self.agent.sample(dist)
            mask = self.agent.get_mask(action)
            masked_action = mask * action
            next_state = self.agent.update_img(state, masked_action)

            final_reward = self.disc(next_state)
            reward = tf.squeeze(final_reward)
            # reward = tf.squeeze(final_reward - tf.squeeze(self.disc(state)))

            exp = _Experience(state, action, reward, mask)

            traj.add_experience(exp)
            state = next_state
            pbar.set_postfix_str(f"Reward: {final_reward}")

            img = tf.squeeze(tf.clip_by_value(state, 0, 1)).numpy()
            img = cv2.resize(img, (img.shape[0] * 5, img.shape[1] * 5), interpolation=cv2.INTER_NEAREST)
            imshow('image', img)
            cv2.waitKey(1)

        pbar.close()
        traj.end()
        self.buffer.add_trajectory(traj)
        self.fake_buffer.add(tf.clip_by_value(state, 0, 1))
        self._train_step()
        self.buffer.reset()

        return final_reward

    def _train_step(self):
        with tf.GradientTape() as tape:
            states = self.buffer.state_buffer
            actions = self.buffer.action_buffer
            rewards = self.buffer.reward_buffer
            masks = self.buffer.mask_buffer

            dist, critic = self.agent.call(states)
            log_probs = self.agent.log_prob(dist, actions)
            log_probs *= masks

            critic = tf.squeeze(critic)
            adv = rewards - critic

            log_probs = tf.reduce_sum(log_probs, [1, 2, 3])

            actor_loss = -tf.reduce_mean(log_probs * adv)
            critic_loss = tf.reduce_mean(adv**2)
            loss = actor_loss + critic_loss

        grad = tape.gradient(loss, self.agent.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.agent.trainable_variables))

    def run(self):
        for e in range(settings.RLGen.Training.num_episodes):
            r = self._run_episode(e + 1)
            if r > 0.99:
                cv2.waitKey(0)
                return
