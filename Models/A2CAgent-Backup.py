import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import A2C as settings
from DenseNet import dense_block
import numpy as np
import cv2
from utils import imshow
import matplotlib.pyplot as plt

import scipy.signal
from utils import generate_noisy_input, display_images


class A2CAgent:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        # actor
        a = layers.Input(shape=input_shape)
        x = dense_block(a, input_shape[-1], num_layers=6, growth_rate=6, self_attention=False)[0]
        mean = layers.Conv2D(4, 3, activation="tanh", padding="same")(x) * settings.max_action

        self.actor = tf.keras.Model(inputs=a, outputs=mean)

    def generate(self, img=None, steps=None, count=1):
        if img is None:
            img = tf.random.uniform((count, *self.input_shape), 0, 1)
        if steps is None:
            steps = settings.Training.max_episode_length

        for s in range(steps):
            mean = self.call(img)
            action = self.sample(mean, 1)
            img = self.update_img(img, action)

        return img

    def call(self, state):
        mean = self.actor(state)
        return mean

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
        size = settings.Training.update_interval
        # Buffer initialization
        self.state_buffer = np.zeros(
            (size, *state_size), dtype=np.float32
        )

        self.action_buffer = np.zeros((size, *state_size), dtype=np.float32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.log_prob_buffer = np.zeros(size, dtype=np.float32)
        self.gamma = 0.99
        self.lam = 0.95

        self.count = 0

    def add_experience(self, state, action, reward, value, log_prob):
        self.state_buffer[self.count] = state
        self.action_buffer[self.count] = action
        self.reward_buffer[self.count] = reward
        self.value_buffer[self.count] = value
        self.log_prob_buffer[self.count] = log_prob
        self.count += 1

    def end(self, last_value):
        rewards = np.append(self.reward_buffer, last_value)
        values = np.append(self.value_buffer, last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer = self.discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

    @staticmethod
    def discounted_cumulative_sums(x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class A2CTrainer:
    def __init__(self, agent, disc, real):
        self.input_shape = agent.input_shape
        self.real = real
        self.agent = agent
        self.critic = tf.keras.models.clone_model(disc)
        self.disc = disc
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=settings.Training.actor_lr)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=settings.Training.critic_lr)

    def _run_episode(self, pbar):
        state = generate_noisy_input(self.real)

        reward = None
        trajs = [_Trajectory(self.input_shape) for _ in range(settings.Training.num_samples_per_state)]

        for step in range(settings.Training.max_episode_length):
            # step
            action, reward, next_state, value, log_prob = self._step(state)
            for s, a, r, v, p, traj in zip(state, action, reward, value, log_prob, trajs):
                traj.add_experience(s, a, r, v, p)

            state = next_state

            # train
            if (step + 1) % settings.Training.update_interval == 0:
                for traj in trajs:
                    traj.end(v)

                self._train_step(
                    tf.convert_to_tensor(np.concatenate([t.state_buffer for t in trajs]), dtype=tf.float32),
                    tf.convert_to_tensor(np.concatenate([t.action_buffer for t in trajs]), dtype=tf.float32),
                    tf.convert_to_tensor(np.concatenate([t.log_prob_buffer for t in trajs]), dtype=tf.float32),
                    tf.convert_to_tensor(np.concatenate([t.advantage_buffer for t in trajs]), dtype=tf.float32),
                    tf.convert_to_tensor(np.concatenate([t.return_buffer for t in trajs]), dtype=tf.float32))

                trajs = [_Trajectory(self.input_shape) for _ in range(settings.Training.num_samples_per_state)]

            # update displays
            pbar.set_postfix_str(f"Reward: {reward[0].numpy(), np.mean(reward)}")
            pbar.update()
            display_images(state)

        return reward[0].numpy(), np.mean(reward)

    @tf.function
    def _step(self, state):
        mean = self.agent.actor(state)
        action = self.agent.sample(mean, 1)
        next_state = tf.clip_by_value(self.agent.update_img(state, action), 0, 1)

        reward = 1 - tf.abs(tf.squeeze(self.disc(next_state)) - 1)

        log_prob = self.agent.log_normal_pdf(action, mean, 1.)
        log_prob = tf.reduce_mean(log_prob, [-1, -2, -3])
        critic = self.critic(state)
        return action, reward, next_state, critic, log_prob

    @tf.function
    def _train_critic(self, states, rewards):
        # train critic
        with tf.GradientTape() as tape:
            critic_s = tf.squeeze(self.critic(states))
            adv = rewards - critic_s
            critic_loss = tf.reduce_mean(tf.math.square(adv))
        variables = self.critic.trainable_variables
        grad = tape.gradient(critic_loss, variables)
        self.c_opt.apply_gradients(zip(grad, self.critic.trainable_variables))

    @tf.function
    def _train_actor_step(self, states, actions, adv, log_probs_orig):
        # train actor
        with tf.GradientTape() as tape:
            mean = self.agent.actor.call(states)
            log_probs = self.agent.log_normal_pdf(actions, mean, 1.)
            log_probs = tf.reduce_mean(log_probs, [-1, -2, -3])
            ratio = tf.exp(log_probs - log_probs_orig)

            min_advantage = tf.where(
                adv > 0,
                (1 + 0.2) * adv,
                (1 - 0.2) * adv,
            )

            actor_loss = -tf.reduce_mean(
                tf.minimum(ratio * adv, min_advantage)
            )
        variables = self.agent.actor.trainable_variables
        grad = tape.gradient(actor_loss, variables)

        self.a_opt.apply_gradients(zip(grad, self.agent.actor.trainable_variables))

        log_probs = self.agent.log_normal_pdf(actions, mean, 1.)
        log_probs = tf.reduce_mean(log_probs, [-1, -2, -3])

        kl = tf.reduce_mean(
            log_probs_orig
            - log_probs
        )

        return kl

    def _train_step(self, states, actions, log_probs, adv, returns):
        for _ in range(80):
            kl = self._train_actor_step(states, actions, adv, log_probs)
            if kl > 1.5 * .01:
                # Early Stopping
                break
        for _ in range(80):
            self._train_critic(states, returns)

    def run(self, epoch):
        self.critic.set_weights(self.disc.get_weights())
        pbar = tqdm(total=settings.Training.max_episode_length, colour='green')

        scores_1st = []
        scores_avg = []

        for e in range(settings.Training.num_episodes):
            pbar.reset()
            pbar.set_description(f"Episode {epoch}.{e+1}")

            r1, r2 = self._run_episode(pbar)
            scores_1st.append(np.asscalar(r1))
            scores_avg.append(np.asscalar(r2))

            plt.plot([i + 1 for i in range(len(scores_1st))], scores_1st, 'g')
            plt.plot([i + 1 for i in range(len(scores_avg))], scores_avg, 'b')
            plt.title(f'Scores {epoch}')
            plt.savefig(f'plots/rewards_{epoch}.png')

            if tf.math.sigmoid(r1)  > 0.99:
                plt.clf()
                return
