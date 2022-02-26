import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

import settings
from DenseNet import dense_block
import numpy as np
import cv2
from utils import imshow


class RLGen(tf.keras.Model):
    def __init__(self, input_shape):
        super(RLGen, self).__init__()

        self.magnitude = 1

        # actor
        a = layers.Input(shape=input_shape)
        x = dense_block(a, 0)[0]
        mean = layers.Conv2D(4, 3, activation="tanh", padding="same")(x) * self.magnitude

        self.actor = tf.keras.Model(inputs=a, outputs=mean)

        # critic
        self.critic = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(1)
        ])

    def call(self, x, use_mask=True):
        mean = self.actor(x)
        # if use_mask:
        #     mask = tf.where(tf.random.uniform(mean.shape, 0, 1) < 0, 0., 1.)
        #     mean *= mask
        critic = self.critic(x)

        return mean, critic

    @staticmethod
    def sample(m, sd):
        return tf.random.normal(m.shape, m, sd)

    @staticmethod
    def log_normal_pdf(sample, mean, sd):
        var = tf.math.sqrt(sd)
        logvar = tf.math.log(var)

        log2pi = tf.math.log(2. * np.pi)
        return -.5 * ((sample - mean) ** 2. / var + logvar + log2pi)


class RLGenExperience:
    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward


class RLGenTrajectory:
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

        # mean = np.mean(rs)
        # stddev = np.std(rs) + .0001
        # for exp in self.experiences:
        #     exp.reward = (exp.reward - mean) / stddev


class RLGenBuffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size):
        # Buffer initialization
        self.state_buffer = np.zeros(
            (size, *observation_dimensions), dtype=np.float32
        )

        self.action_buffer = np.zeros((size, *observation_dimensions), dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)

        self.pointer = 0

    def add_trajectory(self, traj):
        for exp in traj.experiences:
            self.state_buffer[self.pointer] = exp.state
            self.action_buffer[self.pointer] = exp.action
            self.reward_buffer[self.pointer] = exp.reward

    def reset(self):
        self.pointer = 0


class RLGenTrainer:
    def __init__(self, rlgen, disc, input_shape, train_buffer, fake_buffer):
        self.fake_buffer = fake_buffer
        self.buffer = train_buffer
        self.agent = rlgen
        self.input_shape = input_shape
        self.disc = disc
        self.opt = tf.keras.optimizers.RMSprop(learning_rate=.0001)

    def _run_episode(self, episode_num):
        state = tf.random.uniform([1, *self.input_shape], 0, 1)
        traj = RLGenTrajectory()
        reward = None

        pbar = tqdm(range(settings.RLGen.Training.max_episode_length), f"Episode {episode_num}")
        for _ in pbar:
            mean, _ = self.agent.call(state)
            action = self.agent.sample(mean, 1)
            next_state = state + action / 255 / self.agent.magnitude

            reward = tf.squeeze(self.disc(next_state)) - tf.squeeze(self.disc(state))

            exp = RLGenExperience(state, action, reward)

            traj.add_experience(exp)
            state = next_state
            pbar.set_postfix_str(f"Reward: {reward}")

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

        return reward

    def _train_step(self):
        with tf.GradientTape() as tape:
            states = self.buffer.state_buffer
            actions = self.buffer.action_buffer
            rewards = self.buffer.reward_buffer

            mean, critic = self.agent.call(states)
            log_probs = self.agent.log_normal_pdf(actions, mean, 1.)

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
