# https://github.com/sfujim/TD3/blob/master/TD3.py
import os
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import TD3 as settings
import numpy as np
from utils import generate_noisy_input, display_images, generate_blotched_input
import matplotlib.pyplot as plt
from Discriminator import Discriminator, DiscriminatorSN, Discriminator_V2
from PIL import Image
from unet import UNet
import cv2


# learn encoding rather than hard code since shape and size are always constant
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight("pos embedding", shape=[1, *input_shape[1:]], initializer=tf.keras.initializers.RandomNormal())

    def call(self, x):
        # broadcast
        e = x + self.w - x
        return tf.concat([x, e], -1)


class TD3Agent:
    def __init__(self, input_shape, spectral_norm=False, v2=False):
        self.input_shape = input_shape
        self.spectral_norm = spectral_norm
        self.v2 = v2
        a = layers.Input(shape=input_shape)
        x = a
        x = PositionalEncoding()(x)

        x = UNet(x)

        mean = layers.Conv2D(input_shape[-1], 1, activation="tanh", padding="same", use_bias=False)(x)
        # mean = layers.Conv2D(input_shape[-1], 3, activation="tanh", padding="same", use_bias=False)(x)

        self.actor = tf.keras.Model(inputs=a, outputs=mean)
        self.critic_1 = _Critic(input_shape, spectral_norm, v2)
        self.critic_2 = _Critic(input_shape, spectral_norm, v2)
        self.real = None

    def save(self, save_dir):
        self.actor.save_weights(f'{save_dir}/actor/model')
        self.critic_1.save_weights(f'{save_dir}/critic_1/model')
        self.critic_2.save_weights(f'{save_dir}/critic_2/model')

    def load(self, save_dir):
        self.actor.load_weights(f'{save_dir}/actor/model')
        self.critic_1.load_weights(f'{save_dir}/critic_1/model')
        self.critic_2.load_weights(f'{save_dir}/critic_2/model')

    def call(self, state, training=True):
        return self.actor(state, training)

    def generate(self, img=None, steps=None, count=1, display=False):
        if img is None:
            img = tf.random.uniform((count, *self.input_shape), 0, 1)
        if steps is None:
            steps = settings.Training.max_episode_length

        e = tqdm(range(steps), "Generating") if display else range(steps)
        for s in e:
            img = self.generate_step(img)
            if display:
                display_images(img)

        return img

    @tf.function
    def generate_step(self, img):
        action = self.call(img, training=False) + tf.random.normal(img.shape) * settings.noise
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
    def __init__(self, input_shape, spectral_norm=False, v2=False):
        super(_Critic, self).__init__()
        # self.dn = Discriminator([*input_shape[:-1], input_shape[-1]*2])
        if spectral_norm:
            self.dn = DiscriminatorSN(input_shape)
        elif v2:
            self.dn = Discriminator_V2(input_shape)
        else:
            self.dn = Discriminator(input_shape)

    def call(self, state, action, delta_score):
        if delta_score:
            next = TD3Agent.update_img(state, action, False)
            x = tf.concat([state, next], -1)
        else:
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

    def save(self, save_path):
        np.savez(save_path,
                 state_buffer=self.state_buffer,
                 action_buffer=self.action_buffer,
                 reward_buffer=self.reward_buffer,
                 next_state_buffer=self.next_state_buffer,
                 count=self.count)

    def load(self, save_path):
        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None

        data = np.load(save_path)

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
    def __init__(self, agent, disc, real, save_dir, delta_score=False):
        self.buffer = _Buffer(agent.input_shape)
        self.real = real
        self.agent = agent
        self.disc = disc
        self.delta_score = delta_score
        self.actor_optimizer = tf.keras.optimizers.Adam(.00001)
        self.critic_optimizer = tf.keras.optimizers.Adam(.00001)

        a2 = TD3Agent(agent.input_shape, agent.spectral_norm, agent.v2)
        self.actor_target = a2.actor
        self.critic_1_target = a2.critic_1
        self.critic_2_target = a2.critic_2

        self.actor_target.set_weights(self.agent.actor.get_weights())
        self.critic_1_target.set_weights(self.agent.critic_1.get_weights())
        self.critic_2_target.set_weights(self.agent.critic_2.get_weights())

        print(self.actor_optimizer.lr)
        print(self.critic_optimizer.lr)
        self.update_count = 0
        self.scores_1st = []
        self.scores_avg = []
        self.episode = 0
        self.save_dir = save_dir

    def save(self):
        if not os.path.exists(f'{self.save_dir}/TD3'):
            os.makedirs(f'{self.save_dir}/TD3')
        if not os.path.exists(f'{self.save_dir}/TD3/optimizer'):
            os.makedirs(f'{self.save_dir}/TD3/optimizer')
        np.savez(f'{self.save_dir}/TD3/other', plot_a=self.scores_1st, plot_b=self.scores_avg, count=self.episode)
        self.agent.save(f'{self.save_dir}/TD3/model')
        self.buffer.save(f"{self.save_dir}/TD3/buffer")
        self.actor_target.save_weights(f'{self.save_dir}/TD3/model/actor_target/model')
        self.critic_1_target.save_weights(f'{self.save_dir}/TD3/model/critic_1_target/model')
        self.critic_2_target.save_weights(f'{self.save_dir}/TD3/model/critic_2_target/model')
        a_opt_weights = self.actor_optimizer.get_weights()
        c_opt_weights = self.critic_optimizer.get_weights()
        np.savez(f"{self.save_dir}/TD3/optimizer/actor", *a_opt_weights)
        np.savez(f"{self.save_dir}/TD3/optimizer/critic", *c_opt_weights)

    def load(self):
        if not os.path.exists(f'{self.save_dir}/TD3/optimizer/actor.npz'):
            return
        self.buffer.load(f"{self.save_dir}/TD3/buffer.npz")

        # optimizers
        a_opt_weights_data = np.load(f"{self.save_dir}/TD3/optimizer/actor.npz")
        c_opt_weights_data = np.load(f"{self.save_dir}/TD3/optimizer/critic.npz")

        a_opt_weights = [a_opt_weights_data[x] for x in a_opt_weights_data.files]
        c_opt_weights = [c_opt_weights_data[x] for x in c_opt_weights_data.files]
        states, actions, rewards, next_states = self.buffer.get_init_sample()
        self._train_critic_step(actions, states, rewards, next_states)
        self._train_actor_step(states, states)

        self.actor_optimizer.set_weights(a_opt_weights)
        self.critic_optimizer.set_weights(c_opt_weights)
        a_opt_weights_data.close()
        c_opt_weights_data.close()

        # models - after optimizers
        self.agent.load(f'{self.save_dir}/TD3/model')
        self.actor_target.load_weights(f'{self.save_dir}/TD3/model/actor_target/model')
        self.critic_1_target.load_weights(f'{self.save_dir}/TD3/model/critic_1_target/model')
        self.critic_2_target.load_weights(f'{self.save_dir}/TD3/model/critic_2_target/model')

        others = np.load(f"{self.save_dir}/TD3/other.npz")
        self.scores_1st = list(others['plot_a'])
        self.scores_avg = list(others['plot_b'])
        self.episode = others['count']
        others.close()

    def run(self, epoch):
        im = None
        while self.episode < settings.Training.num_episodes:
            r1, r2, im = self._run_episode(epoch)
            self.scores_1st.append(np.asscalar(r1.numpy()))
            self.scores_avg.append(np.asscalar(r2))
            plt.clf()

            plt.plot([i+1 for i in range(len(self.scores_1st))], self.scores_1st, 'g' if self.episode < 10000 else 'g,')
            plt.plot([i+1 for i in range(len(self.scores_avg))], self.scores_avg, 'b' if self.episode < 10000 else 'b,')
            plt.title(f'Scores {epoch}')
            if not os.path.exists(f'{self.save_dir}/plots'):
                os.mkdir(f'{self.save_dir}/plots')
            plt.savefig(f'{self.save_dir}/plots/rewards_{epoch}.png')
            plt.savefig(f'{self.save_dir}/plots/_rewards_current.png')
            if r1  > 0.95:
                break
            self.episode += 1
            if self.episode % 100 == 0:
                self.save()

        self.episode = 0
        self.buffer.clear()
        self.scores_1st = []
        self.scores_avg = []
        self.save()

        if not os.path.exists(f'{self.save_dir}/generated'):
            os.makedirs(f'{self.save_dir}/generated')
        if im.shape[-1] != 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{self.save_dir}/generated/{epoch}.png", im)

    def _run_episode(self, epoch):
        state = generate_blotched_input(self.real)
        real = state[-1]
        reward = None
        pbar = tqdm(
            range(settings.Training.max_episode_length),
            f"Episode {epoch}.{self.episode}",
            colour=settings.Training.color)

        if not self.delta_score:
            self.buffer.add(real, np.zeros_like(real), 1, real)
        for step in pbar:
            action, next_state, reward, delta = self._next(state)
            rr = delta if self.delta_score else reward
            for s, a, r, n in zip(state, action, rr, next_state):
                self.buffer.add(s, a, r, n)
            state = next_state

            if self.buffer.count >= settings.Training.batch_size:
                states, actions, rewards, next_state = self.buffer.sample()
                self._train_step(actions, states, rewards, next_state, self.real[np.random.choice(len(self.real), settings.Training.batch_size)])

            pbar.set_postfix_str(f"Reward: {reward[0].numpy(), np.mean(reward), reward[-1].numpy()}")
            im = state
            # im = tf.concat([im, self.agent.pos_enc.w], 0)
            if step % 20 == 0:
                display_images(im)
        pbar.close()
        im = display_images(state)
        return reward[0], np.mean(reward), im

    def _train_step(self, actions, states, rewards, next_state, real):
        self._train_critic_step(actions, states, rewards, next_state)

        self.update_count += 1
        if self.update_count % settings.Training.actor_update_interval == 0:
            self._train_actor_step(states, real)
            self._update_targets(settings.tau)

    @tf.function()
    def _next(self, state):
        action = self.agent.actor(state) + tf.random.normal(state.shape) * settings.noise
        next_state = self.agent.update_img(state, action)
        reward = 1 - tf.abs(tf.squeeze(self.disc(next_state, training=False)) - 1)
        r1 = 1 - tf.abs(tf.squeeze(self.disc(state, training=False)) - 1)
        delta = reward - r1
        return action, next_state, reward, delta * 100

    @tf.function()
    def _train_critic_step(self, actions, states, rewards, next_state):
        noise = tf.random.normal(actions.shape) * settings.noise
        next_action = self.actor_target(next_state, training=False)

        target_q1 = tf.squeeze(self.critic_1_target(next_state, next_action, self.delta_score, training=False))
        target_q2 = tf.squeeze(self.critic_2_target(next_state, next_action, self.delta_score, training=False))

        rewards = tf.squeeze(rewards)

        target_q = tf.minimum(target_q1, target_q2)
        target_q = rewards + settings.discount * target_q
        target_q = tf.stop_gradient(target_q)
        with tf.GradientTape() as tape:
            q1 = tf.squeeze(self.agent.critic_1(states, actions, self.delta_score))
            q2 = tf.squeeze(self.agent.critic_2(states, actions, self.delta_score))

            c1_loss = tf.keras.losses.MSE(target_q, q1)
            c2_loss = tf.keras.losses.MSE(target_q, q2)

            c_loss = c1_loss + c2_loss

        trainable_variables = self.agent.critic_1.trainable_variables + self.agent.critic_2.trainable_variables
        grads = tape.gradient(c_loss, trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, trainable_variables))

        return c1_loss

    @tf.function()
    def _train_actor_step(self, states, real):
        with tf.GradientTape() as tape:
            a_loss = -self.agent.critic_1(states, self.agent.actor(states), self.delta_score, training=False)
            a_loss = tf.reduce_mean(a_loss)

            # real_out = self.agent.actor(real)
            # real_loss = tf.reduce_mean(tf.square(real_out))
            # a_loss += real_loss
        grads = tape.gradient(a_loss, self.agent.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.agent.actor.trainable_variables))

    def _update_targets(self, tau):
        for a, b in zip(self.actor_target.weights, self.agent.actor.weights):
            a.assign(b * tau + a * (1 - settings.tau))
        for a, b in zip(self.critic_1_target.weights, self.agent.critic_1.weights):
            a.assign(b * tau + a * (1 - settings.tau))
        for a, b in zip(self.critic_2_target.weights, self.agent.critic_2.weights):
            a.assign(b * tau + a * (1 - settings.tau))
