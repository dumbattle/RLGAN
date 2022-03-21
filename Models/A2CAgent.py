import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import A2C as settings
from DenseNet import dense_block
import numpy as np
import cv2
from utils import imshow
import matplotlib.pyplot as plt


class A2CAgent:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        # actor
        a = layers.Input(shape=input_shape)
        x = dense_block(a, input_shape[-1], num_layers=6, growth_rate=6, self_attention=True)[0]
        mean = layers.Conv2D(4, 3, activation="tanh", padding="same")(x) * settings.max_action

        self.actor = tf.keras.Model(inputs=a, outputs=mean)

    @staticmethod
    def sample(m, sd):
        return tf.random.normal(m.shape, m, sd)

    @staticmethod
    def update_img(image, action):
        return image + action / 255 / settings.max_action

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
            (settings.Training.update_interval, *state_size), dtype=np.float32
        )

        self.action_buffer = np.zeros((settings.Training.update_interval, *state_size), dtype=np.float32)
        self.reward_buffer = np.zeros(settings.Training.update_interval, dtype=np.float32)

        self.count = 0

    def add_experience(self, state, action, reward):
        self.state_buffer[self.count] = state
        self.action_buffer[self.count] = action
        self.reward_buffer[self.count] = reward
        self.count += 1

    def end(self):
        gamma = .9
        reward = self.reward_buffer[-1] / (1 - gamma)
        reward = self.reward_buffer[-1]
        for i in reversed(range(len(self.reward_buffer))):
            reward = reward * gamma + self.reward_buffer[i] * (1 - gamma)
            self.reward_buffer[i] = reward


class A2CTrainer:
    def __init__(self, agent, disc, fake_buffer):
        self.input_shape = agent.input_shape
        self.fake_buffer = fake_buffer
        self.agent = agent
        self.critic = tf.keras.models.clone_model(disc)
        self.disc = disc
        self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=settings.Training.actor_lr)
        self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=settings.Training.critic_lr)

    def _run_episode(self, pbar):
        state = tf.random.uniform([1, *self.input_shape], 0, 1)
        reward = None
        traj = _Trajectory(self.input_shape)

        for step in range(settings.Training.max_episode_length):
            action, reward, next_state = self._step(state)
            traj.add_experience(state, action, reward)

            state = next_state
            pbar.set_postfix_str(f"Reward: {tf.sigmoid(reward)}")
            pbar.update()

            img = tf.squeeze(tf.clip_by_value(state, 0, 1)).numpy()
            img = cv2.resize(img, (img.shape[0] * 5, img.shape[1] * 5), interpolation=cv2.INTER_NEAREST)
            imshow('image', img)
            cv2.waitKey(1)
            if (step + 1) % settings.Training.update_interval == 0:
                traj.end()
                self._train_step(
                    tf.convert_to_tensor(traj.state_buffer),
                    tf.convert_to_tensor(traj.action_buffer),
                    tf.convert_to_tensor(traj.reward_buffer))
                traj = _Trajectory(self.input_shape)

        # only save good images
        if reward > .9:
            self.fake_buffer.add(state)

        return tf.math.sigmoid(reward)

    @tf.function
    def _step(self, state):
        mean = self.agent.actor(state)
        action = self.agent.sample(mean, 1)
        next_state = self.agent.update_img(state, action)

        reward = tf.squeeze(self.disc(next_state))
        return action, reward, next_state

    @tf.function
    def _train_step(self, states, actions, rewards):
        with tf.GradientTape() as tape:
            mean = self.agent.actor.call(states)
            log_probs = self.agent.log_normal_pdf(actions, mean, 1.)
            log_probs = tf.reduce_sum(log_probs, [1, 2, 3])

            critic_s = tf.squeeze(self.critic(states))

            adv = rewards - critic_s

            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(adv))
            critic_loss = tf.reduce_mean(tf.math.square(adv))
            loss = actor_loss + critic_loss

        variables = self.agent.actor.trainable_variables + self.critic.trainable_variables
        grad = tape.gradient(loss, variables)

        self.a_opt.apply_gradients(
            zip(grad[:len(self.agent.actor.trainable_variables)], self.agent.actor.trainable_variables))
        self.c_opt.apply_gradients(
            zip(grad[len(self.agent.actor.trainable_variables):], self.critic.trainable_variables))

    def run(self):
        self.critic.set_weights(self.disc.get_weights())
        pbar = tqdm(total=settings.Training.max_episode_length, colour='green')

        mr = 0

        for e in range(settings.Training.num_episodes):
            pbar.reset()
            pbar.set_description(f"Episode {e+1}")
            r = self._run_episode(pbar)
            mr = mr * .8 + r * .2
            if mr > 0.99:
                return
