class DenseNet:
    num_blocks = 4
    layers_per_block = 4
    growth_rate = 12
    reduction = .5

    activation = 'relu'


class Discriminator:
    class Training:
        batch_size = 32


class RLGen:
    class Training:
        max_episode_length = 300
        num_episodes = 1000


dataset_path = "data/datasets/pkmn_small_ds.npy"
