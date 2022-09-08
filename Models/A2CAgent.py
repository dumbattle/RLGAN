import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import A2C as settings
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
from utils import generate_noisy_input, display_images
import os


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


class A2CAgent:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.pos_enc = PositionalEncoding()
        a = layers.Input(shape=input_shape)
        x = a
        x = self.pos_enc(x)
        x = UNet(x)
        mean = layers.Conv2D(input_shape[-1], 1, activation="tanh", padding="same", use_bias=False)(x) * settings.max_action

        self.actor = tf.keras.Model(inputs=a, outputs=mean)

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
        mean = self.call(img, training=False)
        action = A2CAgent.sample(mean)
        img = self.update_img(img, action)
        return img

    def call(self, state, training=True):
        mean = self.actor(state, training)
        return mean

    @staticmethod
    def sample(m):
        return tf.random.normal(m.shape, m)

    @staticmethod
    def update_img(image, action):
        result = image + action / 255.0 / settings.max_action
        result = tf.clip_by_value(result, 0, 1)
        return result

    @staticmethod
    def log_normal_pdf(sample, mean):
        log2pi = tf.math.log(2. * np.pi)
        return -.5 * ((sample - mean) ** 2. + log2pi)


class _Trajectory:
    def __init__(self, state_size):
        size = settings.Training.max_episode_length
        # Buffer initialization
        self.state_buffer = np.zeros((size, *state_size), dtype=np.float32)
        self.action_buffer = np.zeros((size, *state_size), dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)

        self.gamma = 0.99

        self.count = 0

    def add_experience(self, state, action, reward):
        self.state_buffer[self.count] = state
        self.action_buffer[self.count] = action
        self.reward_buffer[self.count] = reward
        self.count += 1

    def end(self):
        r = self.reward_buffer[-1] / (1 - self.gamma)
        for i in reversed(range(len(self.reward_buffer))):
            r *= self.gamma
            r += self.reward_buffer[i]
            self.reward_buffer[i] = r


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
            action, reward, next_state = self._step(state)
            for s, a, r, traj in zip(state, action, reward, trajs):
                traj.add_experience(s, a, r)
            state = next_state

            # update displays
            pbar.set_postfix_str(f"Reward: {reward[0].numpy(), np.mean(reward)}")
            pbar.update()

            im = state
            im = tf.concat([im, (self.agent.pos_enc.w + 1) / 2], 0)
            display_images(im)

        # train
        for traj in trajs:
            traj.end()
        actor_grads = None
        critic_grads = None
        for b in range(settings.Training.num_batches_per_episode):
            ind_start = b * settings.Training.batchsize
            ind_end = ind_start + settings.Training.batchsize
            ag, cg = self._train_step(
                tf.convert_to_tensor(np.concatenate([t.state_buffer[ind_start:ind_end] for t in trajs]), dtype=tf.float32),
                tf.convert_to_tensor(np.concatenate([t.action_buffer[ind_start:ind_end] for t in trajs]), dtype=tf.float32),
                tf.convert_to_tensor(np.concatenate([t.reward_buffer[ind_start:ind_end] for t in trajs]), dtype=tf.float32)
            )

            if actor_grads is None:
                actor_grads = ag
            else:
                actor_grads = [a + b for a, b in zip(actor_grads, ag)]

            if critic_grads is None:
                critic_grads = cg
            else:
                critic_grads = [a + b for a, b in zip(critic_grads, cg)]
        self.c_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        self.a_opt.apply_gradients(zip(actor_grads, self.agent.actor.trainable_variables))

        return reward[0].numpy(), np.mean(reward)

    @tf.function
    def _step(self, state):
        mean = self.agent.actor(state)
        action = self.agent.sample(mean)
        next_state = self.agent.update_img(state, action)

        reward = 1 - tf.abs(tf.squeeze(self.disc(next_state, training=False)) - 1)
        return action, reward, next_state

    @tf.function
    def _train_critic(self, states, rewards):
        # train critic
        with tf.GradientTape() as tape:
            critic_s = tf.squeeze(self.critic(states))

            adv = rewards - critic_s

            critic_loss = tf.reduce_mean(tf.math.square(adv))
        variables = self.critic.trainable_variables
        grad = tape.gradient(critic_loss, variables)
        # self.c_opt.apply_gradients(zip(grad, self.critic.trainable_variables))
        return grad

    @tf.function
    def _train_actor_step(self, states, actions, reward):
        # train actor
        with tf.GradientTape() as tape:
            mean = self.agent.actor.call(states)
            log_probs = self.agent.log_normal_pdf(actions, mean)
            log_probs = tf.reduce_sum(log_probs, [-1, -2, -3])

            values = tf.squeeze(self.critic(states))
            adv = reward - values

            actor_loss = -tf.reduce_mean(
                log_probs * adv
            )
        variables = self.agent.actor.trainable_variables
        grad = tape.gradient(actor_loss, variables)

        # self.a_opt.apply_gradients(zip(grad, self.agent.actor.trainable_variables))
        return grad

    def _train_step(self, states, actions, reward):
        ag = self._train_actor_step(states, actions, reward)
        cg = self._train_critic(states, reward)
        return ag, cg

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
            if not os.path.exists('plots'):
                os.mkdir('plots')
            plt.savefig(f'plots/rewards_{epoch}.png')

            if tf.math.sigmoid(r1)  > 0.99:
                plt.clf()
                return
