class DenseNet:
    num_blocks = 4
    layers_per_block = 4
    growth_rate = 8
    reduction = .5

    activation = 'relu'


class Discriminator:
    class Training:
        batch_size = 32


class A2C:
    # in pixels
    max_action = 3

    class Training:
        max_episode_length = 150
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
    max_action = .01
    noise = 1e-6

    class Training:
        color = 'blue'
        batch_size = 4
        buffer_size = 10000
        max_episode_length = 100
        num_episodes = 1000


class DiscreteAgent:
    # must be at least 1
    max_action = 1

    class Training:
        max_episode_length = 300
        num_episodes = 1000


dataset_path = "data/datasets/pkmn_small_ds.npy"
