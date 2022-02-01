import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from gans.abstract_gan import AbstractGan
from gans.ns_gan import NonSaturatingGan
from gans.r1_gan import R1Gan
from gans.r1_gan_batch_stats import R1GanBatchStats
from gans.vanilla_gan import VanillaGan
from gans.wgan import WGan
from gans.wgan_gp import WGanGP
from util import evaluation
from util.datasets.one_dim.multi_mode_dataset import MultiModeDataset
from util.datasets.one_dim.multi_normal_dataset import MultiNormalDataset
import matplotlib.pyplot as plt

from util.datasets.one_dim.random_dataset import NormalRandomDataset

h_size = 128
z_size = 64
lr = 5e-5
n_bins = 120
bin_range = (-1.5, 1.5)
cuda = True
output_images=True

algorithms = {
    "GAN": VanillaGan(h_size, z_size, learning_rate=lr),
    "NS-GAN": NonSaturatingGan(h_size, z_size, learning_rate=lr),
    "R1 NS-GAN γ=10": R1Gan(h_size, z_size, learning_rate=lr, gamma=10.0),
    "WGAN": WGan(h_size, z_size, learning_rate=lr),
    # "WGAN (run 2)": WGan(h_size, z_size, learning_rate=lr),
    # "WGAN (run 3)": WGan(h_size, z_size, learning_rate=lr),
    "WGAN-GP": WGanGP(h_size, z_size, learning_rate=lr, lambd=10.0),
    "WGAN-GP 1|1": WGanGP(h_size, z_size, learning_rate=lr, lambd=10.0, G_step_every=1),
    "NS-GAN BN": NonSaturatingGan(h_size, z_size, learning_rate=lr, use_batchnorm=True),
    "R1 NS-GAN BN γ=10": R1Gan(h_size, z_size, learning_rate=lr, gamma=10.0, use_batchnorm=True),
    # "WGAN-GP BN Generator": WGanGP(h_size, z_size, learning_rate=lr, lambd=10.0, use_batchnorm=True),
    # "R1 NS-GAN Batch Stats": R1GanBatchStats(h_size, z_size, learning_rate=lr, gamma=10.0),

}
if cuda:
    for name in algorithms.keys():
        algorithms[name] = algorithms[name].cuda()
# dataset = NormalRandomDataset(mean=0.4, stddev=0.05)
# testset = NormalRandomDataset(mean=0.4, stddev=0.05, size=500)
dataset = MultiNormalDataset(mean1=-0.5, std1=0.3, mean2=0.5, std2=0.2, size=10000)
testset = MultiNormalDataset(mean1=-0.5, std1=0.3, mean2=0.5, std2=0.2, size=500)
# dataset = MultiModeDataset([-1, 0, 0.5, 0.75, 0.875, 1.0], size_per_mode=2000, stddev=0.05)
# testset = MultiModeDataset([-1, 0, 0.5, 0.75, 0.875, 1.0], size_per_mode=100, stddev=0.05)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
js_divergence_values = defaultdict(lambda: [])
step_values = []

steps_taken = 0


def epoch(epoch_number=None, log_values=False, log_ever_n_steps=100):
    global steps_taken
    for real_batch in dataloader:
        if cuda:
            real_batch = real_batch.cuda()

        for name, algorithm in algorithms.items():
            algorithm.train_step(real_batch)

        if steps_taken % log_ever_n_steps == 0:
            step_values.append(steps_taken)
            for name, algorithm in algorithms.items():
                jsd = evaluation.binned_jsd(testset.data, algorithm, n_bins, bin_range=bin_range)
                js_divergence_values[name].append(jsd)

                output_image(name, algorithm, steps_taken)
        steps_taken += 1

    if log_values:
        print("=== EPOCH %d ===" % (epoch_number,))

    if log_values:
        for name, algorithm in algorithms.items():
            jsd = evaluation.binned_jsd(testset.data, algorithm, n_bins, bin_range=bin_range)
            print("%s: %.4f" % (name, jsd))

    if log_values:
        print()


def generate_comparison(name, algorithm: AbstractGan, plot_d=False):
    plt.close("all")
    generated_samples = algorithm.generate_batch(dataset.data.size(0)).view(-1).detach().cpu().numpy()
    real_samples = dataset.data.view(-1).detach().cpu().numpy()
    plt.hist(real_samples, label="Real samples", bins=n_bins, range=bin_range)
    plt.hist(generated_samples, label="Generated samples", bins=n_bins, range=bin_range)
    plt.legend()
    plt.title("Output for %s" % (name,))
    plt.show()

    if plot_d:
        x_values, d_values = algorithm.get_discriminator_values_1d(range=bin_range)
        x_values = x_values.view(-1).detach().cpu().numpy()
        d_values = d_values.view(-1).detach().cpu().numpy()
        plt.plot(x_values, d_values, label="D output")
        plt.title("D Output for %s" % (name,))
        plt.show()

ylims = dict()

def output_image(algorithm_name, algorithm: AbstractGan, step, root="results"):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

    generated_samples = algorithm.generate_batch(dataset.data.size(0)).view(-1).detach().cpu().numpy()
    real_samples = dataset.data.view(-1).detach().cpu().numpy()
    axes[0].hist(real_samples, label="Real samples", bins=n_bins, range=bin_range, density=True)
    axes[0].hist(generated_samples, label="Generated samples", bins=n_bins, range=bin_range , density=True)

    x_values, d_values = algorithm.get_discriminator_values_1d(range=bin_range)
    mi, ma = d_values.min().detach().cpu().item(), d_values.max().detach().cpu().item()
    if algorithm_name in ylims:
        mi = min(ylims[algorithm_name][0], mi)
        ma = max(ylims[algorithm_name][1], ma)
    ylims[algorithm_name] = (mi, ma)

    x_values = x_values.view(-1).detach().cpu().numpy()
    d_values = d_values.view(-1).detach().cpu().numpy()
    axes[1].plot(x_values, d_values, label="D output")
    axes[1].set_ylim(mi, ma)

    plt.legend()
    plt.title("Output for %s on step %d" % (algorithm_name, step))

    if not os.path.exists(os.path.join(root, algorithm_name)):
        os.mkdir(os.path.join(root, algorithm_name))

    plt.savefig(os.path.join(root, algorithm_name, "step-%04d.png" % step))
    plt.close("all")

try:
    for epoch_number in range(70):
        epoch(epoch_number, log_values=True)
except KeyboardInterrupt:
    print("Training interrupted")

for name, algorithm in algorithms.items():
    generate_comparison(name, algorithm, plot_d=True)

# Make comparison plot
plt.title("Jensen-Shannon divergence with real data at step n")
for name, results in js_divergence_values.items():
    plt.plot(step_values, results, label=name)
plt.legend()
plt.xlabel("Training step")
plt.ylabel("JSD(real || generated)")
plt.show()
