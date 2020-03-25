import h5py
from util import generate_batch, create_net

epochs = 50
steps = 200
batch_size = 128


def main():
    model = create_net()

    for i in range(epochs):
        # iteratively add noise during training. Full noise after 66% of the set.
        noise_level = 3*i/epochs
        if noise_level >= 2: noise_level = 2
        batch = generate_batch(batch_size, noise_level)
        model.fit_generator(batch, steps_per_epoch=steps)
        model.save_weights("model.h5")

if __name__ == "__main__":
    main()
