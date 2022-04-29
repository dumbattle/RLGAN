# https://github.com/sfujim/TD3/blob/master/TD3.py
import os
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import TD3 as settings
import numpy as np
from utils import generate_noisy_input, display_images
from DenseNet import DenseNet, dense_block
import matplotlib.pyplot as plt


class TD3Agent:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        a = layers.Input(shape=input_shape)
        x = dense_block(a, input_shape[-1], num_layers=6, growth_rate=6, self_attention=True)[0]
        mean = layers.Conv2D(4, 3, activation="tanh", padding="same")(x) * settings.max_action

        self.actor = tf.keras.Model(inputs=a, outputs=mean)

        self.critic_1 = _Critic(input_shape)
        self.critic_2 = _Critic(input_shape)

    def save(self):
        self.actor.save_weights('saves/TD3/model/actor/model')
        self.critic_1.save_weights('saves/TD3/model/critic_1/model')
        self.critic_2.save_weights('saves/TD3/model/critic_2/model')

    def load(self):
        self.actor.load_weights('saves/TD3/model/actor/model')
        self.critic_1.load_weights('saves/TD3/model/critic_1/model')
        self.critic_2.load_weights('saves/TD3/model/critic_2/model')

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
        self.dn = DenseNet(input_shape)
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(1)

    def call(self, state, action):
        x = TD3Agent.update_img(state, action)
        x = self.dn(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class _Buffer:
    def __init__(self, input_shape):
        self.count = 0

        self.state_buffer = np.zeros((settings.Training.buffer_size, *input_shape), dtype=np.float32)
        self.action_buffer = np.zeros((settings.Training.buffer_size, *input_shape), dtype=np.float32)
        self.reward_buffer = np.zeros(settings.Training.buffer_size, dtype=np.float32)
        self.next_state_buffer = np.zeros((settings.Training.buffer_size, *input_shape), dtype=np.float32)

    def save(self):
        np.savez("saves/TD3/buffer",
                 state_buffer=self.state_buffer,
                 action_buffer=self.action_buffer,
                 reward_buffer=self.reward_buffer,
                 next_state_buffer=self.next_state_buffer,
                 count=self.count)

    def load(self):
        if not os.path.exists("saves/TD3/buffer.npz"):
            return

        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None

        data = np.load("saves/TD3/buffer.npz")

        self.state_buffer = data['state_buffer']
        self.action_buffer = data['action_buffer']
        self.reward_buffer = data['reward_buffer']
        self.next_state_buffer = data['next_state_buffer']
        self.count = data['count']

        data.close()

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

    def get_init_sample(self):
        indices = [0, 0]
        states = tf.convert_to_tensor(self.state_buffer[indices], dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_buffer[indices], dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_buffer[indices], dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.next_state_buffer[indices], dtype=tf.float32)

        return states, actions, rewards, next_states

    def clear(self):
        self.count = 0


class TD3Trainer:
    def __init__(self, agent, disc, real):
        self.buffer = _Buffer(agent.input_shape)
        self.real = real
        self.agent = agent
        self.disc = disc

        self.actor_optimizer = tf.keras.optimizers.RMSprop(.0001)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(.0001)

        self.actor_target = tf.keras.models.clone_model(agent.actor)
        self.critic_1_target = _Critic(agent.input_shape)
        self.critic_2_target = _Critic(agent.input_shape)

        self.critic_1_target.set_weights(self.agent.critic_1.get_weights())
        self.critic_2_target.set_weights(self.agent.critic_2.get_weights())

        self.update_count = 0
        self.scores_1st = []
        self.scores_avg = []
        self.episode = 0

    def save(self):
        print('Saving...')
        np.savez("saves/TD3/other", plot_a=self.scores_1st, plot_b=self.scores_avg, count=self.episode)
        print('Agent...')
        self.agent.save()
        print('Buffer...')
        self.buffer.save()
        print('Targets...')
        self.actor_target.save_weights('saves/TD3/model/actor_target/model')
        self.critic_1_target.save_weights('saves/TD3/model/critic_1_target/model')
        self.critic_2_target.save_weights('saves/TD3/model/critic_2_target/model')
        print('Optimizers...')
        a_opt_weights = self.actor_optimizer.get_weights()
        c_opt_weights = self.critic_optimizer.get_weights()
        np.savez("saves/TD3/optimizer/actor", *a_opt_weights)
        np.savez("saves/TD3/optimizer/critic", *c_opt_weights)

    def load(self):
        if not os.path.exists('saves/TD3/optimizer/actor.npz'):
            return
        self.buffer.load()

        # optimizers
        a_opt_weights_data = np.load("saves/TD3/optimizer/actor.npz")
        c_opt_weights_data = np.load("saves/TD3/optimizer/critic.npz")

        a_opt_weights = [a_opt_weights_data[x] for x in a_opt_weights_data.files]
        c_opt_weights = [c_opt_weights_data[x] for x in c_opt_weights_data.files]
        states, actions, rewards, next_states = self.buffer.get_init_sample()
        self._train_critic_step(actions, states, rewards, next_states)
        self._train_actor_step(states, 1)

        self.actor_optimizer.set_weights(a_opt_weights)
        self.critic_optimizer.set_weights(c_opt_weights)
        a_opt_weights_data.close()
        c_opt_weights_data.close()

        # models - after optimizers
        self.agent.load()
        self.actor_target.load_weights('saves/TD3/model/actor_target/model')
        self.critic_1_target.load_weights('saves/TD3/model/critic_1_target/model')
        self.critic_2_target.load_weights('saves/TD3/model/critic_2_target/model')

        others = np.load("saves/TD3/other.npz")
        self.scores_1st = list(others['plot_a'])
        self.scores_avg = list(others['plot_b'])
        self.episode = others['count']
        others.close()

    def run(self, epoch):
        while self.episode < settings.Training.num_episodes:
            r1, r2 = self._run_episode(epoch)
            self.scores_1st.append(np.asscalar(r1.numpy()))
            self.scores_avg.append(np.asscalar(r2))
            plt.plot([i+1 for i in range(len(self.scores_1st))], self.scores_1st, 'g')
            plt.plot([i+1 for i in range(len(self.scores_avg))], self.scores_avg, 'b')
            plt.title(f'Scores {epoch}')
            plt.savefig(f'plots/rewards_{epoch}.png')
            self.episode += 1
            if tf.math.sigmoid(r1)  > 0.99:
                break
            if self.episode % 100 == 0:
                self.save()

        plt.clf()
        self.episode = 0
        self.buffer.clear()
        self.scores_1st = []
        self.scores_avg = []
        self.save()

    def _run_episode(self, epoch):
        state = generate_noisy_input(self.real)
        reward = None
        total_c_loss = 0
        pbar = tqdm(
            range(settings.Training.max_episode_length),
            f"Episode {epoch}.{self.episode}",
            colour=settings.Training.color)

        for step in pbar:
            action, next_state, reward = self._next(state)

            for s, a, r, n in zip(state, action, reward, next_state):
                self.buffer.add(s, a, r, n)
            state = next_state

            if self.buffer.count >= settings.Training.batch_size:
                states, actions, rewards, next_state = self.buffer.sample()
                total_c_loss += self._train_step(actions, states, rewards, next_state).numpy()

            pbar.set_postfix_str(f"Reward: {reward[0].numpy(), np.mean(reward)} C-Loss: {total_c_loss / (step + 1)}")
            display_images(state)
        pbar.close()

        return reward[0], np.mean(reward)

    def _train_step(self, actions, states, rewards, next_state):
        c_loss = self._train_critic_step(actions, states, rewards, next_state)

        self.update_count += 1
        if self.update_count % settings.Training.actor_update_interval == 0:
            self._train_actor_step(states, c_loss)

        return c_loss

    @tf.function
    def _next(self, state):
        action = tf.clip_by_value(
            self.agent.actor(state) + tf.random.normal(state.shape, 0, settings.noise),
            -settings.max_action,
            settings.max_action
        )
        # no noise needed for exploration
        # exploration is accomplished by starting with real-ish images
        # action = self.agent.actor(state)
        next_state = self.agent.update_img(state, action)
        reward = tf.squeeze(self.disc(next_state))
        # reward = 5 - tf.math.abs(5 - reward)  # max reward at 5 - discourage super high rewards
        return action, next_state, reward

    @tf.function
    def _train_critic_step(self, actions, states, rewards, next_state):
        next_action = tf.clip_by_value(self.actor_target(next_state), -settings.max_action, settings.max_action)

        target_q1 = tf.squeeze(self.critic_1_target(next_state, next_action))
        target_q2 = tf.squeeze(self.critic_2_target(next_state, next_action))

        rewards = tf.squeeze(rewards)

        target_q = tf.minimum(target_q1, target_q2)
        target_q = rewards + settings.discount * target_q

        with tf.GradientTape() as tape:
            q1 = tf.squeeze(self.agent.critic_1(states, actions))
            q2 = tf.squeeze(self.agent.critic_2(states, actions))

            c1_loss = tf.keras.losses.MSE(target_q, q1)
            c2_loss = tf.keras.losses.MSE(target_q, q2)

            c_loss = c1_loss + c2_loss

        trainable_variables = self.agent.critic_1.trainable_variables + self.agent.critic_2.trainable_variables
        grads = tape.gradient(c_loss, trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, trainable_variables))

        return c1_loss

    @tf.function
    def _train_actor_step(self, states, critic_loss):
        with tf.GradientTape() as tape:
            a_loss = -self.agent.critic_1(states, self.agent.actor(states))
            a_loss = tf.reduce_mean(a_loss)

        grads = tape.gradient(a_loss, self.agent.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.agent.actor.trainable_variables))

        for a, b in zip(self.actor_target.weights, self.agent.actor.weights):
            a.assign(b * settings.tau + a * (1 - settings.tau))
        for a, b in zip(self.critic_1_target.weights, self.agent.critic_1.weights):
            a.assign(b * settings.tau + a * (1 - settings.tau))
        for a, b in zip(self.critic_2_target.weights, self.agent.critic_2.weights):
            a.assign(b * settings.tau + a * (1 - settings.tau))
