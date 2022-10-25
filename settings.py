class TD3:
    max_action = 2
    noise = .1

    discount = 0.9
    tau = .005

    class Training:
        color = 'blue'
        batch_size = 32
        buffer_size = 30000
        max_episode_length = 150
        num_episodes = 100000

        actor_update_interval = 2


class PixelAE:
    palette_code_size = 8
    palette_size = 16


class DenseNet:
    num_blocks = 4
    self_attention = [3]
    layers_per_block = 6
    growth_rate = 8
    reduction = .5

    activation = 'softplus'


class Discriminator:
    size = 10000
    new_size = 1000

    class Training:
        batch_size = 32


class A2C:
    # in pixels
    max_action = 3

    class Training:
        batchsize = 32
        num_samples_per_state = 8
        num_batches_per_episode = 8
        max_episode_length = batchsize * num_batches_per_episode
        actor_lr = 1e-5
        critic_lr = 1e-5
        num_episodes = 10000
        gamma = .9


class A2C_D:
    class Training:
        batchsize = 32
        num_samples_per_state = 8
        num_batches_per_episode = 8
        max_episode_length = batchsize * num_batches_per_episode
        actor_lr = 2e-4
        critic_lr = 2e-4
        num_episodes = 10000
        gamma = .9


class AC:
    class Training:
        max_episode_length = 100
        num_episodes = 1000


class DDPGAgent:
    max_action = .01
    discount = .9

    class Training:
        color = 'blue'
        batch_size = 64
        buffer_size = 10000
        max_episode_length = 100
        num_episodes = 1000


class SACAgent:
    max_action = 3
    noise = 1e-6

    class Training:
        color = 'blue'
        batch_size = 16
        buffer_size = 5000
        max_episode_length = 150
        num_episodes = 1000


class DiscreteAgent:
    # must be at least 1
    max_action = 1

    class Training:
        max_episode_length = 300
        num_episodes = 1000


dataset_path = "data/datasets/pkmn_small_ds.npy"
