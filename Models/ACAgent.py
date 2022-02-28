import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import AC as settings
from DenseNet import dense_block
import numpy as np
import cv2
from utils import imshow


class ACAgent:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        # actor
        a = layers.Input(shape=input_shape)
        x = dense_block(a, 0)[0]
        mean = layers.Conv2D(4, 3, activation="tanh", padding="same")(x) * settings.max_action

        self.actor = tf.keras.Model(inputs=a, outputs=mean)

        # critic
        a = layers.Input(shape=input_shape)
        x = dense_block(a, 0)[0]
        c = layers.GlobalAveragePooling2D()(x)
        c = layers.Dense(1)(c)
        self.critic = tf.keras.Model(inputs=a, outputs=c)

    @staticmethod
    def sample(m, sd):
        return tf.random.normal(m.shape, m, sd) / 255 / settings.max_action

    @staticmethod
    def log_normal_pdf(sample, mean, sd):
        var = tf.math.sqrt(sd)
        log_var = tf.math.log(var)

        log2pi = tf.math.log(2. * np.pi)
        return -.5 * ((sample - mean) ** 2. / var + log_var + log2pi)


class _Trajectory:
    def __init__(self, state_size):
        # Buffer initialization
        self.state_buffer = np.zeros(
            (settings.Training.max_episode_length, *state_size), dtype=np.float32
        )

        self.action_buffer = np.zeros((settings.Training.max_episode_length, *state_size), dtype=np.float32)
        self.reward_buffer = np.zeros(settings.Training.max_episode_length, dtype=np.float32)

        self.count = 0

    def add_experience(self, state, action, reward):
        self.state_buffer[self.count] = state
        self.action_buffer[self.count] = action
        self.reward_buffer[self.count] = reward

    def end(self):
        gamma = .9
        reward = self.reward_buffer[-1]
        for i in reversed(range(len(self.reward_buffer))):
            reward = reward * gamma + self.reward_buffer[i] * (1 - gamma)
            self.reward_buffer[i] = reward


class ACTrainer:
    def __init__(self, agent, disc, fake_buffer):
        self.input_shape = agent.input_shape
        self.fake_buffer = fake_buffer
        self.agent = agent
        self.disc = disc
        self.opt = tf.keras.optimizers.RMSprop(learning_rate=.0001)

    def _run_episode(self, episode_num):
        state = tf.random.uniform([1, *self.input_shape], 0, 1)
        reward = None
        traj = _Trajectory(self.input_shape)
        pbar = tqdm(range(settings.Training.max_episode_length), f"Episode {episode_num}")
        for _ in pbar:
            mean = self.agent.actor(state)
            action = self.agent.sample(mean, 1)
            next_state = state + action

            reward = tf.squeeze(self.disc(next_state))

            traj.add_experience(state, action, reward)
            state = next_state
            pbar.set_postfix_str(f"Reward: {tf.math.sigmoid(reward)}")

            img = tf.squeeze(tf.clip_by_value(state, 0, 1)).numpy()
            img = cv2.resize(img, (img.shape[0] * 5, img.shape[1] * 5), interpolation=cv2.INTER_NEAREST)
            imshow('image', img)
            cv2.waitKey(1)
        traj.end()
        self._train_step(
            tf.convert_to_tensor(traj.state_buffer),
            tf.convert_to_tensor(traj.action_buffer),
            tf.convert_to_tensor(traj.reward_buffer))
        pbar.close()
        self.fake_buffer.add(tf.clip_by_value(state, 0, 1))

        return tf.math.sigmoid(reward)

    def _train_step(self, states, actions, rewards):
        with tf.GradientTape() as tape:
            mean = self.agent.actor.call(states)
            log_probs = self.agent.log_normal_pdf(actions, mean, 1.)

            critic_s = tf.squeeze(self.agent.critic(states))

            adv = rewards - critic_s

            log_probs = tf.reduce_sum(log_probs, [1, 2, 3])

            actor_loss = -tf.reduce_mean(log_probs * adv)
            critic_loss = tf.reduce_mean(adv**2)
            loss = actor_loss + critic_loss

        variables = self.agent.actor.trainable_variables + self.agent.critic.trainable_variables
        grad = tape.gradient(loss, variables)
        self.opt.apply_gradients(zip(grad, variables))

    def run(self):
        for e in range(settings.Training.num_episodes):
            r = self._run_episode(e + 1)
            if r > 0.99:
                return
