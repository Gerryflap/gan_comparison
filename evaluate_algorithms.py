from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from gans.abstract_gan import AbstractGan
from gans.ns_gan import NonSaturatingGan
from gans.vanilla_gan import VanillaGan
from util import evaluation
from util.multi_mode_dataset import MultiModeDataset
from util.multi_normal_dataset import MultiNormalDataset
from util.random_dataset import NormalRandomDataset
import matplotlib.pyplot as plt

h_size = 64
z_size = 12
lr = 1e-4
n_bins = 40
bin_range = (-4, 4)
cuda = True

algorithms = {
    "GAN": VanillaGan(h_size, z_size, learning_rate=lr),
    "NS-GAN": NonSaturatingGan(h_size, z_size, learning_rate=lr)
}
if cuda:
    for name in algorithms.keys():
        algorithms[name] = algorithms[name].cuda()

# dataset = MultiNormalDataset(mean1=1.5, std1=0.8, mean2=-1.5, std2=0.5)
dataset = MultiModeDataset([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], size_per_mode=2000, stddev=0.2)
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
                jsd = evaluation.binned_jsd(dataset.data, algorithm, n_bins, bin_range=bin_range)
                js_divergence_values[name].append(jsd)
        steps_taken += 1

    if log_values:
        print("=== EPOCH %d ===" % (epoch_number,))

    if log_values:
        for name, algorithm in algorithms.items():
            jsd = evaluation.binned_jsd(dataset.data, algorithm, n_bins, bin_range=bin_range)
            print("%s: %.4f" % (name, jsd))

    if log_values:
        print()


def generate_comparison(name, algorithm: AbstractGan):
    generated_samples = algorithm.generate_batch(dataset.data.size(0)).view(-1).detach().cpu().numpy()
    real_samples = dataset.data.view(-1).detach().cpu().numpy()
    plt.hist(real_samples, label="Real samples", bins=n_bins, range=bin_range)
    plt.hist(generated_samples, label="Generated samples", bins=n_bins, range=bin_range)
    plt.legend()
    plt.title("Output for %s" % (name,))
    plt.show()


for epoch_number in range(100):
    epoch(epoch_number, log_values=True)

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
