import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import A2C_D as settings
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
from utils import generate_noisy_input, display_images
import os
from PIL import Image


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


class A2CDAgent:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.pos_enc = PositionalEncoding()
        a = layers.Input(shape=input_shape)
        x = a
        x = self.pos_enc(x)
        x = UNet(x)
        x = layers.Conv2D(input_shape[-1] * 3, 1, padding="same", use_bias=False)(x)
        x = tf.reshape(x, [-1, *x.shape[1:-1], input_shape[-1], 3])

        self.actor = tf.keras.Model(inputs=a, outputs=x)

    def generate(self, img=None, steps=None, count=1, display=False):
        if img is None:
            img = tf.random.uniform((count, *self.input_shape), 0, 1)
        if steps is None:
            steps = settings.Training.max_episode_length

        for s in range(steps):
            img = self.generate_step(img)
            if display:
                display_images(img)

        return img

    @tf.function
    def generate_step(self, img):
        mean = self.call(img, training=False)
        action = self.sample(mean)
        img = self.update_img(img, action)
        return img

    def call(self, state, training=True):
        logits = self.actor(state, training)
        return logits

    def sample(self, m):
        shape = m.shape
        x = tf.reshape(m, (-1, 3))
        x = tf.random.categorical(x, 1)
        x = tf.reshape(x, shape[:-1])
        return x

    @staticmethod
    def update_img(image, action):
        action = tf.cast(action, tf.float32)
        action -= 1
        result = image + action / 100.0
        result = tf.clip_by_value(result, 0, 1)
        return result

    @staticmethod
    def log_prob(sample, logits):
        s = sample.shape
        sample = tf.reshape(sample, (-1,))
        logits = tf.reshape(logits, (-1, logits.shape[-1]))
        probs = tf.math.softmax(logits)
        sample = tf.one_hot(sample, 3)
        x = probs * sample
        x = tf.reduce_sum(x, -1)
        x = tf.reshape(x, s)
        return tf.math.log(x)

    def save(self, save_dir):
        self.actor.save_weights(f'{save_dir}/actor/model')

    def load(self, save_dir):
        self.actor.load_weights(f'{save_dir}/actor/model')


class _Trajectory:
    def __init__(self, state_size):
        size = settings.Training.max_episode_length
        # Buffer initialization
        self.state_buffer = np.zeros((size, *state_size), dtype=np.float32)
        self.action_buffer = np.zeros((size, *state_size), dtype=np.int64)
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


class A2CDTrainer:
    def __init__(self, agent, disc, real, save_dir):
        self.input_shape = agent.input_shape
        self.real = real
        self.agent = agent
        self.critic = tf.keras.models.clone_model(disc)
        self.disc = disc
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=settings.Training.actor_lr)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=settings.Training.critic_lr)
        self.save_dir = save_dir
        self.episode = 0
        self.scores_1st = []
        self.scores_avg = []

    def save(self):
        if not os.path.exists(f'{self.save_dir}/A2C'):
            os.makedirs(f'{self.save_dir}/A2C')
        if not os.path.exists(f'{self.save_dir}/A2C/optimizer'):
            os.makedirs(f'{self.save_dir}/A2C/optimizer')
        # Actor
        self.agent.save(f'{self.save_dir}/A2C/actor/model')

        # Critic
        self.critic.save_weights(f'{self.save_dir}/A2C/critic/model')

        # Optimizers
        a_opt_weights = self.a_opt.get_weights()
        c_opt_weights = self.c_opt.get_weights()
        np.savez(f"{self.save_dir}/A2C/optimizer/actor", *a_opt_weights)
        np.savez(f"{self.save_dir}/A2C/optimizer/critic", *c_opt_weights)

        # plots?
        np.savez(f'{self.save_dir}/A2C/other', plot_a=self.scores_1st, plot_b=self.scores_avg, count=self.episode)

    def load(self):
        if not os.path.exists(f'{self.save_dir}/A2C/optimizer/actor.npz'):
            return
        # optimizers
        a_opt_weights_data = np.load(f"{self.save_dir}/A2C/optimizer/actor.npz")
        c_opt_weights_data = np.load(f"{self.save_dir}/A2C/optimizer/critic.npz")

        a_opt_weights = [a_opt_weights_data[x] for x in a_opt_weights_data.files]
        c_opt_weights = [c_opt_weights_data[x] for x in c_opt_weights_data.files]

        states = tf.random.normal([2, *self.input_shape])
        actions = tf.random.uniform([2, *self.input_shape], 0, 3, dtype=tf.int64)
        rewards = tf.random.normal([2, 1])
        ag, cg = self._train_step(states, actions, rewards)

        self.a_opt.apply_gradients(zip(ag, self.agent.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(cg, self.critic.trainable_variables))
        self.a_opt.set_weights(a_opt_weights)
        self.c_opt.set_weights(c_opt_weights)
        a_opt_weights_data.close()
        c_opt_weights_data.close()

        # models - after optimizers
        self.agent.load(f'{self.save_dir}/A2C/actor/model')
        self.critic.load_weights(f'{self.save_dir}/A2C/critic/model')

        others = np.load(f"{self.save_dir}/A2C/other.npz")
        self.scores_1st = list(others['plot_a'])
        self.scores_avg = list(others['plot_b'])
        self.episode = others['count']
        others.close()

    def _run_episode(self, pbar):
        state = generate_noisy_input(self.real)

        reward = None
        trajs = [_Trajectory(self.input_shape) for _ in range(settings.Training.num_samples_per_state)]
        for step in range(settings.Training.max_episode_length):
            # step
            action, reward, next_state, rf = self._step(state)
            for s, a, r, traj in zip(state, action, reward, trajs):
                traj.add_experience(s, a, r)
            state = next_state

            # update displays
            pbar.set_postfix_str(f"Reward: {rf[0].numpy(), np.mean(rf)}")
            pbar.update()

            im = state
            # im = tf.concat([im, (self.agent.pos_enc.w + 1) / 2], 0)
            display_images(im)

        # train
        for traj in trajs:
            traj.end()
        actor_grads = None
        critic_grads = None
        for b in range(settings.Training.num_batches_per_episode-1):
            ind_start = b * settings.Training.batchsize
            ind_end = ind_start + settings.Training.batchsize
            ag, cg = self._train_step(
                tf.convert_to_tensor(np.concatenate([t.state_buffer[ind_start:ind_end] for t in trajs]), dtype=tf.float32),
                tf.convert_to_tensor(np.concatenate([t.action_buffer[ind_start:ind_end] for t in trajs]), dtype=tf.int64),
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

        return rf[0].numpy(), np.mean(rf), im

    @tf.function
    def _step(self, state):
        mean = self.agent.call(state)
        action = self.agent.sample(mean)
        next_state = self.agent.update_img(state, action)

        reward_s = 1 - tf.abs(tf.squeeze(self.disc(state, training=False)) - 1)
        reward_n = 1 - tf.abs(tf.squeeze(self.disc(next_state, training=False)) - 1)
        reward = reward_n - reward_s
        return action, reward, next_state, reward_n

    @tf.function
    def _train_critic(self, states, rewards):
        # train critic
        with tf.GradientTape() as tape:
            critic_s = tf.squeeze(self.critic(states))
            adv = critic_s - rewards
            critic_loss = tf.reduce_mean(tf.math.square(adv))
        variables = self.critic.trainable_variables
        grad = tape.gradient(critic_loss, variables)
        # self.c_opt.apply_gradients(zip(grad, variables))
        return grad

    @tf.function
    def _train_actor_step(self, states, actions, reward):
        # train actor
        with tf.GradientTape() as tape:
            logits = self.agent.actor.call(states)
            log_probs = self.agent.log_prob(actions, logits)
            log_probs = tf.reduce_sum(log_probs, [-1, -2, -3])

            values = tf.squeeze(self.critic(states))
            adv = reward - values

            actor_loss = -tf.reduce_mean(
                log_probs * tf.stop_gradient(adv)
            )
        variables = self.agent.actor.trainable_variables
        grad = tape.gradient(actor_loss, variables)

        # self.a_opt.apply_gradients(zip(grad, variables))
        return grad

    def _train_step(self, states, actions, reward):
        ag = self._train_actor_step(states, actions, reward)
        cg = self._train_critic(states, reward)
        return ag, cg

    def run(self, epoch):
        self.critic.set_weights(self.disc.get_weights())
        pbar = tqdm(total=settings.Training.max_episode_length, colour='green')

        while self.episode < settings.Training.num_episodes:
            pbar.reset()
            pbar.set_description(f"Episode {epoch}.{self.episode+1}")

            r1, r2, im = self._run_episode(pbar)
            self.scores_1st.append(np.asscalar(r1))
            self.scores_avg.append(np.asscalar(r2))

            plt.plot([i + 1 for i in range(len(self.scores_1st))], self.scores_1st, 'g')
            plt.plot([i + 1 for i in range(len(self.scores_avg))], self.scores_avg, 'b')
            plt.title(f'Scores {epoch}')

            if not os.path.exists(f'{self.save_dir}/plots'):
                os.mkdir(f'{self.save_dir}/plots')
            plt.savefig(f'{self.save_dir}/plots/rewards_{epoch}.png')
            plt.savefig(f'{self.save_dir}/plots/_rewards_current.png')

            if tf.math.sigmoid(r1)  > 0.99:
                plt.clf()
                break
            self.episode += 1
            if self.episode % 100 == 0:
                self.save()

        self.episode = 0
        self.scores_1st = []
        self.scores_avg = []
        self.save()

        if not os.path.exists(f'{self.save_dir}/generated'):
            os.makedirs(f'{self.save_dir}/generated')

        im = Image.fromarray(im)
        im.save(f"{self.save_dir}/generated/{epoch}.png")


if __name__ == '__main__':
    print(A2CDAgent.log_prob(tf.constant([1, 2, 0]), tf.constant([[4., 6., 1.], [1., 2., 3], [4., 5., 6]])))
