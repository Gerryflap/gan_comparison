from collections import defaultdict

from torch.utils.data import DataLoader

from gans.abstract_gan import AbstractGan
from gans.ns_gan import NonSaturatingGan
from gans.r1_gan import R1Gan
from gans.r1_wgan import R1WGan
from gans.vanilla_gan import VanillaGan
from gans.wgan import WGan
from util import evaluation
from util.datasets.two_dim.multi_normal_dataset import MultiNormalDataset2D
import matplotlib.pyplot as plt

h_size = 64
z_size = 32
lr = 3e-4
n_bins = 40
bin_range = (-4, 4)
cuda = True

algorithms = {
    "GAN": VanillaGan(h_size, z_size, n_features=2, learning_rate=lr),
    "NS-GAN": NonSaturatingGan(h_size, z_size, n_features=2, learning_rate=lr),
    "R1 NS-GAN γ=2": R1Gan(h_size, z_size, n_features=2, learning_rate=lr, gamma=2.0),
    "R1 WGAN  γ=2": R1WGan(h_size, z_size, n_features=2, learning_rate=lr, gamma=2.0),
    "WGAN": WGan(h_size, z_size, n_features=2, learning_rate=lr)
}
if cuda:
    for name in algorithms.keys():
        algorithms[name] = algorithms[name].cuda()

dataset = MultiNormalDataset2D(mean1=(1.5, 1.5), std1=0.8, mean2=(-1.5, -1.5), std2=0.8)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
js_divergence_values = defaultdict(lambda: [])
step_values = []

steps_taken = 0


def epoch(epoch_number=None, log_values=False, log_ever_n_steps=100):
    global steps_taken
    for real_batch in dataloader:
        for name, algorithm in algorithms.items():
            algorithm.train_step(real_batch)

        if steps_taken % log_ever_n_steps == 0:
            step_values.append(steps_taken)
            for name, algorithm in algorithms.items():
                jsd = evaluation.binned_jsd_2d(dataset.data, algorithm, n_bins, bin_range=bin_range)
                js_divergence_values[name].append(jsd)
        steps_taken += 1

    if log_values:
        print("=== EPOCH %d ===" % (epoch_number,))

    if log_values:
        for name, algorithm in algorithms.items():
            jsd = evaluation.binned_jsd_2d(dataset.data, algorithm, n_bins, bin_range=bin_range)
            print("%s: %.4f" % (name, jsd))

    if log_values:
        print()


def generate_comparison(name, algorithm: AbstractGan):
    generated_samples = algorithm.generate_batch(dataset.data.size(0)).detach().cpu().numpy()
    real_samples = dataset.data.detach().cpu().numpy()
    plt.scatter(real_samples[:, 0], real_samples[:, 1], label="Real samples", vmin=bin_range[0], vmax=bin_range[1])
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], label="Generated samples", vmin=bin_range[0], vmax=bin_range[1])
    plt.legend()
    plt.title("Output for %s" % (name,))
    plt.show()


try:
    for epoch_number in range(100):
        epoch(epoch_number, log_values=True, log_ever_n_steps=400)
except KeyboardInterrupt:
    print("Training interrupted")

for name, algorithm in algorithms.items():
    generate_comparison(name, algorithm)

# Make comparison plot
plt.title("Jensen-Shannon divergence with real data at step n")
for name, results in js_divergence_values.items():
    plt.plot(step_values, results, label=name)
plt.legend()
plt.xlabel("Training step")
plt.ylabel("JSD(real || generated)")
plt.show()
