import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from settings import A2C as settings
from DenseNet import dense_block
import numpy as np
import cv2
from utils import imshow
import matplotlib.pyplot as plt

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
        self.reward_buffer = np.zeros(size, dtype=np.float32)

        self.count = 0

    def add_experience(self, state, action, reward):
        self.state_buffer[self.count] = state
        self.action_buffer[self.count] = action
        self.reward_buffer[self.count] = reward
        self.count += 1

    def end(self):
        reward = self.reward_buffer[-1]

        for i in reversed(range(len(self.reward_buffer))):
            reward = reward + self.reward_buffer[i] * settings.Training.gamma
            self.reward_buffer[i] = reward


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

            # train
            if (step + 1) % settings.Training.update_interval == 0:
                for traj in trajs:
                    traj.end()

                self._train_step(
                    tf.convert_to_tensor(np.concatenate([t.state_buffer for t in trajs])),
                    tf.convert_to_tensor(np.concatenate([t.action_buffer for t in trajs])),
                    tf.convert_to_tensor(np.concatenate([t.reward_buffer for t in trajs])))

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

        reward = tf.squeeze(self.disc(next_state))
        return action, reward, next_state

    @tf.function
    def _train_step(self, states, actions, rewards):
        rewards = tf.squeeze(rewards)

        with tf.GradientTape() as tape:
            mean = self.agent.actor.call(states)
            log_probs = self.agent.log_normal_pdf(actions, mean, 1.)
            log_probs = tf.reduce_mean(log_probs, [-1, -2, -3])

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
