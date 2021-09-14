from collections import defaultdict

from torch.utils.data import DataLoader

from gans.abstract_gan import AbstractGan
from gans.ns_gan import NonSaturatingGan
from gans.r1_gan import R1Gan
from gans.r1_gan_batch_stats import R1GanBatchStats
from gans.vanilla_gan import VanillaGan
from gans.wgan import WGan
from gans.wgan_gp import WGanGP
from util import evaluation
from util.datasets.two_dim.circle_of_gaussians_dataset import CircleOfGaussiansDataset2D
from util.datasets.two_dim.multi_normal_dataset import MultiNormalDataset2D
import matplotlib.pyplot as plt

h_size = 64
z_size = 32
lr = 1e-4
n_bins = 20
bin_range = (-2, 2)
cuda = True

algorithms = {
    "NS-GAN": NonSaturatingGan(h_size, z_size, n_features=2, learning_rate=lr),
    "R1 NS-GAN γ=2": R1Gan(h_size, z_size, n_features=2, learning_rate=lr, gamma=2.0),
    "R1 NS-GAN γ=10": R1Gan(h_size, z_size, n_features=2, learning_rate=lr),
    "WGAN": WGan(h_size, z_size, n_features=2, learning_rate=lr),
    "WGAN-GP": WGanGP(h_size, z_size, n_features=2, learning_rate=lr),
    # "NS-GAN BN": NonSaturatingGan(h_size, z_size, n_features=2, learning_rate=lr, use_batchnorm=True),
    # "R1 NS-GAN BN γ=2": R1Gan(h_size, z_size, n_features=2, learning_rate=lr, gamma=2.0, use_batchnorm=True),
    "R1 NS-GAN γ=10 Batch Stats": R1GanBatchStats(h_size, z_size, n_features=2, learning_rate=lr, gamma=10.0),
}
if cuda:
    for name in algorithms.keys():
        algorithms[name] = algorithms[name].cuda()

# dataset = MultiNormalDataset2D(mean1=(1.5, 1.5), std1=0.5, mean2=(-1.5, -1.5), std2=0.5)
# testset = MultiNormalDataset2D(mean1=(1.5, 1.5), std1=0.5, mean2=(-1.5, -1.5), std2=0.5, size=200)
dataset = CircleOfGaussiansDataset2D()
testset = CircleOfGaussiansDataset2D(size=200)
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
                jsd = evaluation.binned_jsd_2d(testset.data, algorithm, n_bins, bin_range=bin_range)
                js_divergence_values[name].append(jsd)
        steps_taken += 1

    if log_values:
        print("=== EPOCH %d ===" % (epoch_number,))

    if log_values:
        for name, algorithm in algorithms.items():
            jsd = evaluation.binned_jsd_2d(testset.data, algorithm, n_bins, bin_range=bin_range)
            print("%s: %.4f" % (name, jsd))

    if log_values:
        print()


def generate_comparison(name, algorithm: AbstractGan):
    generated_samples = algorithm.generate_batch(dataset.data.size(0)).detach().cpu().numpy()
    real_samples = dataset.data.detach().cpu().numpy()
    plt.scatter(real_samples[:, 0], real_samples[:, 1], label="Real samples", vmin=bin_range[0], vmax=bin_range[1], s=2)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], label="Generated samples", vmin=bin_range[0],
                vmax=bin_range[1], s=2)
    plt.legend()
    plt.title("Output for %s" % (name,))
    plt.show()


try:
    for epoch_number in range(500):
        epoch(epoch_number, log_values=True, log_ever_n_steps=100)
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
