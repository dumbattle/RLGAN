import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import DDPGAgent as settings
from DenseNet import dense_block
import numpy as np
import cv2
from utils import imshow


class DDPGAgent(tf.keras.Model):
    def __init__(self, input_shape):
        super(DDPGAgent, self).__init__()
        self.inp_shape = input_shape

        a = layers.Input(shape=input_shape)
        x = dense_block(a, 0)[0]
        x = layers.Conv2D(4, 1, activation="tanh", padding="same")(x) * settings.max_action

        self.actor = tf.keras.Model(inputs=a, outputs=x)

    def call(self, x):
        return self.actor(x)

    @staticmethod
    def update_img(img, action):
        return img + action


class _Buffer:
    def __init__(self, shape):
        self.buffer_counter = 0

        self.state_buffer = np.zeros((settings.Training.buffer_size, *shape))
        self.action_buffer = np.zeros((settings.Training.buffer_size, *shape))
        self.reward_buffer = np.zeros((settings.Training.buffer_size, 1))

        self.critic_optimizer = tf.keras.optimizers.Adam(.002)
        self.actor_optimizer = tf.keras.optimizers.Adam(.001)

    def add_exp(self, state, action, reward):
        index = self.buffer_counter % settings.Training.buffer_size

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.buffer_counter += 1

    def sample(self):
        # Get sampling range
        max_ind = min(self.buffer_counter, settings.Training.buffer_size)

        # Randomly sample indices
        indices = np.random.choice(max_ind, settings.Training.batch_size)
        states = tf.convert_to_tensor(self.state_buffer[indices], dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_buffer[indices], dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_buffer[indices], dtype=tf.float32)

        return states, actions, rewards

    def clear(self):
        self.buffer_counter = 0


class DDPGAgentTrainer:
    def __init__(self, agent, disc, disc_buffer):
        self.input_shape = agent.inp_shape
        self.buffer = _Buffer(agent.inp_shape)
        self.disc = disc

        self.fake_buffer = disc_buffer

        self.actor = agent

        self.target_actor = DDPGAgent(self.input_shape)
        self.target_actor.set_weights(agent.get_weights())
        self.critic = tf.keras.models.clone_model(self.disc)
        self.target_critic = tf.keras.models.clone_model(self.disc)

        self.c_opt = tf.keras.optimizers.Adam(.002)
        self.a_opt = tf.keras.optimizers.Adam(.001)

    def _run_episode(self, episode_num):
        state = tf.random.uniform([1, *self.input_shape], 0, 1)
        reward = None

        pbar = tqdm(
            range(settings.Training.max_episode_length),
            f"Episode {episode_num}",
            colour=settings.Training.color)

        for _ in pbar:
            action = self.actor(state)
            action += tf.random.uniform(action.shape, -settings.max_action / 3, settings.max_action / 3)
            next_state = self.actor.update_img(state, action)

            reward = self.disc(next_state)
            self.buffer.add_exp(state, action, reward)

            state = next_state

            if self.buffer.buffer_counter >= settings.Training.batch_size:
                actions, states, rewards = self.buffer.sample()
                self._train_step(actions, states, rewards)

            # display info
            pbar.set_postfix_str(f"Reward: {reward}")

            img = tf.squeeze(tf.clip_by_value(state, 0, 1)).numpy()
            img = cv2.resize(img, (img.shape[0] * 5, img.shape[1] * 5), interpolation=cv2.INTER_NEAREST)
            imshow('image', img)
            cv2.waitKey(1)
            if reward > .99:
                self.fake_buffer.add(tf.clip_by_value(state, 0, 1))
        pbar.close()
        self.fake_buffer.add(state)
        return reward

    @tf.function
    def _train_step(self, state, action, reward):
        with tf.GradientTape() as tape:
            next_state = self.actor.update_img(state, action)
            next_action = self.target_actor.call(next_state)
            next_next_state = self.actor.update_img(next_state, next_action)
            next_value = self.target_critic(next_next_state)

            target_value = reward + settings.discount * next_value
            critic_value = self.critic(next_state)
            loss = tf.math.reduce_mean(tf.math.square(target_value - critic_value))
        critic_grad = tape.gradient(loss, self.critic.trainable_variables)

        with tf.GradientTape() as tape:
            new_actions = self.actor(state)
            next_state = self.actor.update_img(state, new_actions)
            value = self.critic(next_state)
            loss = -tf.math.reduce_mean(value)
        actor_grad = tape.gradient(loss, self.actor.trainable_variables)

        self.a_opt.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
        self.c_opt.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

    def run(self):
        self.buffer.clear()
        self.critic.set_weights(self.disc.get_weights())
        self.target_critic.set_weights(self.disc.get_weights())

        for episode in range(settings.Training.num_episodes):
            r = self._run_episode(episode + 1)
            if r > 0.99:
                return
