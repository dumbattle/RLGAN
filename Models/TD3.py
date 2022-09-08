# https://github.com/sfujim/TD3/blob/master/TD3.py
import os
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import TD3 as settings
import numpy as np
from utils import generate_noisy_input, display_images
import matplotlib.pyplot as plt
from Discriminator import Discriminator
from PIL import Image
from unet import UNet


# learn encoding rather than hard code since shape and size are always constant
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight("pos embedding", shape=[1, *input_shape[1:]])

    def call(self, x):
        # broadcast
        e = x + self.w - x

        return tf.concat([x, e], -1)


class TD3Agent:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.pos_enc = PositionalEncoding()
        a = layers.Input(shape=input_shape)
        x = a
        x = self.pos_enc(x)
        x = UNet(x)
        mean = layers.Conv2D(input_shape[-1], 1, activation="tanh", padding="same", use_bias=False)(x)

        self.actor = tf.keras.Model(inputs=a, outputs=mean)
        self.critic_1 = _Critic(input_shape)
        self.critic_2 = _Critic(input_shape)

    def save(self):
        if not os.path.exists('saves/TD3/model'):
            os.makedirs('saves/TD3/model')
        self.actor.save_weights('saves/TD3/model/actor/model')
        self.critic_1.save_weights('saves/TD3/model/critic_1/model')
        self.critic_2.save_weights('saves/TD3/model/critic_2/model')

    def load(self):
        self.actor.load_weights('saves/TD3/model/actor/model')
        self.critic_1.load_weights('saves/TD3/model/critic_1/model')
        self.critic_2.load_weights('saves/TD3/model/critic_2/model')

    def call(self, state, training=True):
        return self.actor(state, training)

    def generate(self, img=None, steps=None, count=1):
        if img is None:
            img = tf.random.uniform((count, *self.input_shape), 0, 1)
        if steps is None:
            steps = settings.Training.max_episode_length

        for s in range(steps):
            img = self.generate_step(img)

        return img

    @tf.function
    def generate_step(self, img):
        action = self.call(img, training=False)
        img = self.update_img(img, action)
        return img

    @staticmethod
    def sample(m):
        return m

    @staticmethod
    def update_img(image, action, clip=True):
        result = image + action * settings.max_action / 255.0
        if clip:
            result = tf.clip_by_value(result, 0, 1)
        return result


class _Critic(tf.keras.Model):
    def __init__(self, input_shape):
        super(_Critic, self).__init__()
        self.dn = Discriminator(input_shape)

    def call(self, state, action):
        x = TD3Agent.update_img(state, action, False)
        x = self.dn(x)
        return x


class _Buffer:
    def __init__(self, input_shape):
        self.count = 0

        self.state_buffer = np.zeros((settings.Training.buffer_size, *input_shape), dtype=np.float32)
        self.action_buffer = np.zeros((settings.Training.buffer_size, *input_shape), dtype=np.float32)
        self.reward_buffer = np.zeros(settings.Training.buffer_size, dtype=np.float32)
        self.next_state_buffer = np.zeros((settings.Training.buffer_size, *input_shape), dtype=np.float32)

    def save(self):

        if not os.path.exists('saves/TD3'):
            os.makedirs('saves/TD3')
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

        self.actor_optimizer = tf.keras.optimizers.Adam(.00001)
        self.critic_optimizer = tf.keras.optimizers.Adam(.00001)

        a2 = TD3Agent(agent.input_shape)
        self.actor_target = a2.actor
        self.critic_1_target = a2.critic_1
        self.critic_2_target = a2.critic_2

        self.actor_target.set_weights(self.agent.actor.get_weights())
        self.critic_1_target.set_weights(self.agent.critic_1.get_weights())
        self.critic_2_target.set_weights(self.agent.critic_2.get_weights())

        self.update_count = 0
        self.scores_1st = []
        self.scores_avg = []
        self.episode = 0

    def save(self):
        if not os.path.exists('saves/TD3/model'):
            os.makedirs('saves/TD3/model')
        if not os.path.exists('saves/TD3/optimizer'):
            os.makedirs('saves/TD3/optimizer')
        np.savez("saves/TD3/other", plot_a=self.scores_1st, plot_b=self.scores_avg, count=self.episode)
        self.agent.save()
        self.buffer.save()
        self.actor_target.save_weights('saves/TD3/model/actor_target/model')
        self.critic_1_target.save_weights('saves/TD3/model/critic_1_target/model')
        self.critic_2_target.save_weights('saves/TD3/model/critic_2_target/model')
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
        self._train_actor_step(states)

        self.actor_optimizer.set_weights(a_opt_weights)
        self.critic_optimizer.set_weights(c_opt_weights)
        a_opt_weights_data.close()
        c_opt_weights_data.close()

        # models - after optimizers
        self.agent.load()
        self.actor_target.load_weights('saves/TD3/model/actor_target/model')
        self.critic_1_target.load_weights('saves/TD3/model/critic_1_target/model')
        self.critic_2_target.load_weights('saves/TD3/model/critic_2_target/model')
        print(self.actor_optimizer.lr)
        print(self.critic_optimizer.lr)

        others = np.load("saves/TD3/other.npz")
        self.scores_1st = list(others['plot_a'])
        self.scores_avg = list(others['plot_b'])
        self.episode = others['count']
        others.close()

    def run(self, epoch):
        while self.episode < settings.Training.num_episodes:
            r1, r2, im = self._run_episode(epoch)
            self.scores_1st.append(np.asscalar(r1.numpy()))
            self.scores_avg.append(np.asscalar(r2))
            plt.clf()

            plt.plot([i+1 for i in range(len(self.scores_1st))], self.scores_1st, 'g' if self.episode < 10000 else 'g,')
            plt.plot([i+1 for i in range(len(self.scores_avg))], self.scores_avg, 'b' if self.episode < 10000 else 'b,')
            plt.title(f'Scores {epoch}')
            if not os.path.exists('plots'):
                os.mkdir('plots')
            plt.savefig(f'plots/rewards_{epoch}.png')
            plt.savefig(f'plots/_rewards_current.png')
            if r1  > 0.95:
                break
            self.episode += 1
            if self.episode % 100 == 0:
                # self._update_targets(1)
                self.save()

        self.episode = 0
        self.buffer.clear()
        self.scores_1st = []
        self.scores_avg = []
        self.save()

        if not os.path.exists('generated/TD3'):
            os.makedirs('generated/TD3')

        im = Image.fromarray(im)
        im.save(f"generated/TD3/{epoch}.png")

    def _run_episode(self, epoch):
        state = generate_noisy_input(self.real)
        real = state[1]
        reward = None
        total_c_loss = 0
        pbar = tqdm(
            range(settings.Training.max_episode_length),
            f"Episode {epoch}.{self.episode}",
            colour=settings.Training.color)
        self.buffer.add(real, np.zeros_like(real), 1, real)

        for step in pbar:
            action, next_state, reward = self._next(state)

            for s, a, r, n in zip(state, action, reward, next_state):
                self.buffer.add(s, a, r, n)

            state = next_state

            if self.buffer.count >= settings.Training.batch_size:
                states, actions, rewards, next_state = self.buffer.sample()
                total_c_loss += self._train_step(actions, states, rewards, next_state).numpy()

            pbar.set_postfix_str(f"Reward: {reward[0].numpy(), np.mean(reward), reward[1].numpy()} C-Loss: {total_c_loss / (step + 1)}")
            im = state
            im = tf.concat([im, self.agent.pos_enc.w], 0)
            im = display_images(im)
        pbar.close()

        return reward[0], np.mean(reward), im

    def _train_step(self, actions, states, rewards, next_state):
        c_loss = self._train_critic_step(actions, states, rewards, next_state)

        self.update_count += 1
        if self.update_count % settings.Training.actor_update_interval == 0:
            self._train_actor_step(states)

        return c_loss

    @tf.function()
    def _next(self, state):
        action = self.agent.actor(state)  # + tf.random.normal(state.shape) * settings.noise
        next_state = self.agent.update_img(state, action)
        reward = 1 - tf.abs(tf.squeeze(self.disc(next_state, training=False)) - 1)
        return action, next_state, reward

    @tf.function()
    def _train_critic_step(self, actions, states, rewards, next_state):
        noise = tf.random.normal(actions.shape) * settings.noise
        next_action = self.actor_target(next_state, training=False) + noise

        target_q1 = tf.squeeze(self.critic_1_target(next_state, next_action, training=False))
        target_q2 = tf.squeeze(self.critic_2_target(next_state, next_action, training=False))

        rewards = tf.squeeze(rewards)

        target_q = tf.minimum(target_q1, target_q2)
        target_q = rewards + settings.discount * target_q
        target_q = tf.stop_gradient(target_q)
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

    @tf.function()
    def _train_actor_step(self, states):
        with tf.GradientTape() as tape:
            a_loss = -self.agent.critic_1(states, self.agent.actor(states), training=False)
            a_loss = tf.reduce_mean(a_loss)

        grads = tape.gradient(a_loss, self.agent.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.agent.actor.trainable_variables))
        self._update_targets(settings.tau)

    def _update_targets(self, tau):
        for a, b in zip(self.actor_target.weights, self.agent.actor.weights):
            a.assign(b * tau + a * (1 - settings.tau))
        for a, b in zip(self.critic_1_target.weights, self.agent.critic_1.weights):
            a.assign(b * tau + a * (1 - settings.tau))
        for a, b in zip(self.critic_2_target.weights, self.agent.critic_2.weights):
            a.assign(b * tau + a * (1 - settings.tau))
