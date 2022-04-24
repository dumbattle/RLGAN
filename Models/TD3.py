# https://github.com/sfujim/TD3/blob/master/TD3.py

import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import TD3 as settings
from DenseNet import dense_block
import numpy as np
import cv2
from utils import imshow, generate_noisy_input, display_images
from DenseNet import DenseNet, dense_block
import matplotlib.pyplot as plt


class TD3Agent:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        a = layers.Input(shape=input_shape)
        x = dense_block(a, input_shape[-1], num_layers=6, growth_rate=6, self_attention=False)[0]
        mean = layers.Conv2D(4, 3, activation="tanh", padding="same")(x) * settings.max_action

        self.actor = tf.keras.Model(inputs=a, outputs=mean)

        self.critic_1 = _Critic(input_shape)
        self.critic_2 = _Critic(input_shape)

    def call(self, state):
        return self.actor(state)

    def generate(self, img=None, steps=None, count=1):
        if img is None:
            img = tf.random.uniform((count, *self.input_shape), 0, 1)
        if steps is None:
            steps = settings.Training.max_episode_length

        for s in range(steps):
            action = self.call(img)
            img = self.update_img(img, action)

        return img

    @staticmethod
    def sample(m, sd):
        return m

    @staticmethod
    def update_img(image, action):
        return image + action / 255 / settings.max_action


class _Critic(tf.keras.Model):
    def __init__(self, input_shape):
        super(_Critic, self).__init__()
        input_shape = list(input_shape)
        input_shape[-1] *= 2
        self.dn = DenseNet(input_shape)
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat((state, action / settings.max_action), -1)
        x = self.dn(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class _Buffer:
    def __init__(self, input_shape):
        self.count = 0

        self.state_buffer = np.zeros((settings.Training.buffer_size, *input_shape))
        self.action_buffer = np.zeros((settings.Training.buffer_size, *input_shape))
        self.reward_buffer = np.zeros(settings.Training.buffer_size)
        self.next_state_buffer = np.zeros((settings.Training.buffer_size, *input_shape))

    def add(self, state, action, reward, next_state):
        index = self.count % settings.Training.buffer_size
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state

        self.count += 1

    def sample(self):
        # Get sampling range
        max_ind = min(self.count, settings.Training.buffer_size)
        indices = np.random.choice(max_ind, settings.Training.batch_size)

        # Randomly sample indices
        states = tf.convert_to_tensor(self.state_buffer[indices], dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_buffer[indices], dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_buffer[indices], dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.next_state_buffer[indices], dtype=tf.float32)

        return states, actions, rewards, next_states

    def clear(self):
        self.count = 0


class TD3Trainer:
    def __init__(self, agent, disc, real):
        self.run_count = 0
        self.buffer = _Buffer(agent.input_shape)
        self.real = real
        self.agent = agent
        self.disc = disc

        self.actor_optimizer = tf.keras.optimizers.RMSprop(.0003)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(.0003)

        self.actor_target = tf.keras.models.clone_model(agent.actor)
        self.critic_1_target = _Critic(agent.input_shape)
        self.critic_2_target = _Critic(agent.input_shape)

        self.critic_1_target.set_weights(self.agent.critic_1.get_weights())
        self.critic_2_target.set_weights(self.agent.critic_2.get_weights())

        self.update_count = 0

    def run(self):
        self.run_count += 1
        self.buffer.clear()
        scores_1st = []
        scores_avg = []
        for episode in range(settings.Training.num_episodes):
            r1, r2 = self._run_episode(episode + 1)
            scores_1st.append(np.asscalar(r1.numpy()))
            scores_avg.append(np.asscalar(r2))
            plt.plot([i+1 for i in range(len(scores_1st))], scores_1st, 'g')
            plt.plot([i+1 for i in range(len(scores_avg))], scores_avg, 'b')
            plt.title(f'Scores {self.run_count}')
            plt.savefig(f'plots/rewards_{self.run_count}.png')

            if tf.math.sigmoid(r1)  > 0.99:
                plt.clf()
                return

    def _run_episode(self, episode_num):
        state = generate_noisy_input(self.real)
        reward = None
        pbar = tqdm(
            range(settings.Training.max_episode_length),
            f"Episode {episode_num}",
            colour=settings.Training.color)

        for _ in pbar:
            action, next_state, reward = self._next(state)
            state = next_state

            for s, a, r, n in zip(state, action, reward, next_state):
                self.buffer.add(s, a, r, n)

            if self.buffer.count >= settings.Training.batch_size:
                self.update_count += 1
                actions, states, rewards, next_state = self.buffer.sample()
                self._train_step(actions, states, rewards, next_state)

            pbar.set_postfix_str(f"Reward: {reward[0].numpy(), np.mean(reward)}")
            display_images(state)
        pbar.close()

        return reward[0], np.mean(reward)

    def _train_step(self, actions, states, rewards, next_state):
        # Once image is realistic enough, not need to keep rewarding agent
        r = tf.minimum(rewards, tf.sigmoid(rewards))
        self._train_critic_step(actions, states, r, next_state)

        if self.update_count % settings.Training.actor_update_interval == 0:
            self._train_actor_step(states)

    @tf.function
    def _next(self, state):
        # action = tf.clip_by_value(
        #     self.agent.actor(state) + tf.random.normal(state.shape, 0, settings.noise),
        #     -settings.max_action,
        #     settings.max_action
        # )
        action = self.agent.actor(state)
        next_state = self.agent.update_img(state, action)
        reward = tf.squeeze(self.disc(next_state))
        return action, next_state, reward

    @tf.function
    def _train_critic_step(self, actions, states, rewards, next_state):
        noise = tf.clip_by_value(tf.random.normal(actions.shape) * 0.2, -0.5, 0.5)
        next_action = tf.clip_by_value(self.actor_target(states) + noise, -settings.max_action, settings.max_action)

        target_q1 = self.critic_1_target(next_state, next_action)
        target_q2 = self.critic_2_target(next_state, next_action)

        target_q = tf.minimum(target_q1, target_q2)
        target_q = rewards + settings.discount * target_q
        target_q = tf.stop_gradient(target_q)  # not sure if this line in needed

        with tf.GradientTape() as tape:
            q1 = self.agent.critic_1(next_state, next_action)
            q2 = self.agent.critic_2(next_state, next_action)
            c_loss = tf.keras.losses.MSE(q1, target_q) + tf.keras.losses.MSE(q2, target_q)

        trainable_variables = self.agent.critic_1.trainable_variables + self.agent.critic_2.trainable_variables
        grads = tape.gradient(c_loss, trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, trainable_variables))

    @tf.function
    def _train_actor_step(self, states):
        with tf.GradientTape() as tape:
            a_loss = -self.agent.critic_1(states, self.agent.actor(states))
            a_loss = tf.reduce_mean(a_loss)
        grads = tape.gradient(a_loss, self.agent.actor.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.agent.actor.trainable_variables))

        for a, b in zip(self.actor_target.weights, self.agent.actor.weights):
            a.assign(b * settings.tau + a * (1 - settings.tau))
        for a, b in zip(self.critic_1_target.weights, self.agent.critic_1.weights):
            a.assign(b * settings.tau + a * (1 - settings.tau))
        for a, b in zip(self.critic_2_target.weights, self.agent.critic_2.weights):
            a.assign(b * settings.tau + a * (1 - settings.tau))
