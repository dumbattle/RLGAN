"""
https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC/tf2
"""
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import SACAgent as settings
from DenseNet import dense_block
import numpy as np
import cv2
from utils import imshow
from DenseNet import DenseNet, dense_block
import matplotlib.pyplot as plt


class SACAgent:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        a = layers.Input(shape=input_shape)
        x = dense_block(a, input_shape[-1], num_layers=6, growth_rate=6, self_attention=False)[0]
        mean = layers.Conv2D(4, 3, activation="tanh", padding="same")(x) * settings.max_action

        self.actor = tf.keras.Model(inputs=a, outputs=mean)

    def call(self, state):
        mean = self.actor(state)
        return mean

    def generate(self, img=None, steps=None, count=1):
        if img is None:
            img = tf.random.uniform((count, *self.input_shape), 0, 1)
        if steps is None:
            steps = settings.Training.max_episode_length

        for s in range(steps):
            img = self._generate_step(img)

        return img

    @tf.function
    def _generate_step(self, img):
        mean = self.call(img)
        action = self.sample(mean, 1)
        return self.update_img(img, action)

    @staticmethod
    def sample(m, sd):
        return tf.random.normal(m.shape, m, sd)

    @staticmethod
    def update_img(image, action):
        return image + action / 255 / settings.max_action

    @staticmethod
    def log_prob(sample, mean, sd):
        var = tf.math.sqrt(sd)
        log_var = tf.math.log(var)

        log2pi = tf.math.log(2. * np.pi)
        return -.5 * ((sample - mean) ** 2. / var + log_var + log2pi)


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


class SACTrainer:
    def __init__(self, agent, disc, real):
        self.input_shape = agent.input_shape
        self.disc = disc
        self.buffer = _Buffer(agent.input_shape)

        self.actor = agent
        self.critic_1 = _Critic(agent.input_shape)
        self.critic_2 = _Critic(agent.input_shape)
        self.value_net = tf.keras.models.clone_model(disc)
        self.target_value_net = tf.keras.models.clone_model(disc)

        self.a_opt = tf.keras.optimizers.Adam(.0001)
        self.c_opt = tf.keras.optimizers.Adam(.0002)
        self.v_opt = tf.keras.optimizers.Adam(.0002)

        self.run_count = 0
        self.real = real

    def run(self, epoch):
        self.run_count += 1
        self.buffer.clear()
        self.value_net.set_weights(self.disc.get_weights())
        self.target_value_net.set_weights(self.disc.get_weights())
        scores_1st = []
        scores_avg = []
        for episode in range(settings.Training.num_episodes):
            r1, r2 = self._run_episode(episode + 1, epoch)
            scores_1st.append(np.asscalar(r1.numpy()))
            scores_avg.append(np.asscalar(r2))
            plt.plot([i+1 for i in range(len(scores_1st))], scores_1st, 'g')
            plt.plot([i+1 for i in range(len(scores_avg))], scores_avg, 'b')
            plt.title(f'Scores {self.run_count}')
            plt.savefig(f'plots/rewards_{self.run_count}.png')

            if tf.math.sigmoid(r1)  > 0.99:
                plt.clf()
                return

    def _run_episode(self, episode_num, epoch):
        num_real = 8 - 1
        random_samples = self.real[np.random.choice(self.real.shape[0], num_real)]
        state = np.random.uniform(0, 1, [1 + num_real, *self.input_shape])

        for i in range(num_real):
            state[i + 1] = np.where(
                np.random.uniform(0, 1, self.input_shape) < float(i) / num_real,
                state[i+1],
                random_samples[i]
            )
        reward = None

        pbar = tqdm(
            range(settings.Training.max_episode_length),
            f"Episode {epoch}.{episode_num}",
            colour=settings.Training.color)

        for _ in pbar:
            mean = self.actor.call(state)
            action = self.actor.sample(mean, 1)
            next_state = self.actor.update_img(state, action)

            reward = tf.squeeze(self.disc(next_state))

            for s, a, r, n in zip(state, action, reward, next_state):
                self.buffer.add(s, a, tf.minimum(r, 10.0), n)

            state = next_state
            if self.buffer.count >= settings.Training.batch_size:
                actions, states, rewards, next_state = self.buffer.sample()
                self._train_step(actions, states, rewards, next_state)

            # display info
            pbar.set_postfix_str(f"Reward: {reward[0].numpy(), np.mean(reward)}")

            # Display all
            pow2val = 1
            pow2 = 0

            while state.shape[0] > pow2val:
                pow2val *= 2
                pow2 += 1

            img = None
            row = None
            row_count = 0

            for i, im in enumerate(state):
                im = tf.squeeze(tf.clip_by_value(im, 0, 1)).numpy()

                if row is None:
                    row = im
                    row_count = 1
                else:
                    row = cv2.hconcat([row, im])
                    row_count += 1

                if row_count == pow2:
                    if img is None:
                        img = row
                    else:
                        img = cv2.vconcat([img, row])

                    row_count = 0
                    row = None
            if row_count != 0:
                im = np.zeros_like(tf.squeeze(state[0]))

                while row_count < pow2:
                    row = cv2.hconcat([row, im])
                    row_count += 1
                if img is None:
                    img = row
                else:
                    img = cv2.vconcat([img, row])

            img = cv2.resize(img, (img.shape[0] * 5, img.shape[1] * 5), interpolation=cv2.INTER_NEAREST)
            imshow('image', img)
            cv2.waitKey(1)
        pbar.close()
        return reward[0], np.mean(reward)

    @tf.function
    def _train_step(self, state, action, reward, next_state):
        reward = tf.squeeze(reward)
        # value net Training
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value_net(state))

            # next state
            mean = self.actor.call(state)
            new_actions = self.actor.sample(mean, 1)

            # entropy (?)
            log_prob = self.actor.log_prob(new_actions, mean, 1.0)
            log_prob = tf.reduce_mean(log_prob, [-1, -2, -3])

            critic1 = self.critic_1.call(state, new_actions)
            critic2 = self.critic_2.call(state, new_actions)
            critic_val = tf.squeeze(tf.math.minimum(critic1, critic2))
            target_value = critic_val - log_prob

            value_loss = tf.reduce_mean((value - target_value)**2)
        value_grads = tape.gradient(value_loss, self.value_net.trainable_variables)
        self.v_opt.apply_gradients(zip(value_grads, self.value_net.trainable_variables))

        # actor training
        with tf.GradientTape() as tape:
            mean = self.actor.call(state)
            new_actions = self.actor.sample(mean, 1)

            log_prob = self.actor.log_prob(new_actions, mean, 1.0)
            log_prob = tf.reduce_mean(log_prob, [-1, -2, -3])

            critic1 = self.critic_1.call(state, new_actions)
            critic2 = self.critic_2.call(state, new_actions)
            critic_val = tf.squeeze(tf.math.minimum(critic1, critic2))

            actor_loss = tf.reduce_mean(log_prob - critic_val)

        actor_grads = tape.gradient(actor_loss, self.actor.actor.trainable_variables)
        self.a_opt.apply_gradients(zip(actor_grads, self.actor.actor.trainable_variables))

        # critic training
        with tf.GradientTape() as tape:
            next_value = tf.squeeze(self.target_value_net(next_state))

            qhat = 2 * reward + .9 * next_value

            c1 = tf.squeeze(self.critic_1.call(state, action))
            c2 = tf.squeeze(self.critic_2.call(state, action))

            c1_loss = tf.reduce_mean((c1 - qhat)**2)
            c2_loss = tf.reduce_mean((c2 - qhat)**2)

            c12_loss = tf.reduce_mean(c1_loss + c2_loss)

        critic_grads = tape.gradient(c12_loss, self.critic_1.trainable_variables + self.critic_2.trainable_variables)
        self.c_opt.apply_gradients(zip(critic_grads, self.critic_1.trainable_variables + self.critic_2.trainable_variables))

        # update target value
        tau = .005

        for (a, b) in zip(self.target_value_net.weights, self.value_net.weights):
            a.assign(b * tau + a * (1 - tau))

        return value_loss, actor_loss, c12_loss
