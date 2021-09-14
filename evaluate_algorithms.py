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
from util.datasets.one_dim.multi_mode_dataset import MultiModeDataset
from util.datasets.one_dim.multi_normal_dataset import MultiNormalDataset
import matplotlib.pyplot as plt

from util.datasets.one_dim.random_dataset import NormalRandomDataset

h_size = 128
z_size = 64
lr = 5e-5
n_bins = 120
bin_range = (-2, 2)
cuda = True

algorithms = {
    "GAN": VanillaGan(h_size, z_size, learning_rate=lr),
    "NS-GAN": NonSaturatingGan(h_size, z_size, learning_rate=lr),
    "R1 NS-GAN γ=10": R1Gan(h_size, z_size, learning_rate=lr, gamma=10.0),
    "WGAN": WGan(h_size, z_size, learning_rate=lr),
    "WGAN-GP": WGanGP(h_size, z_size, learning_rate=lr, lambd=10.0),
    "NS-GAN BN": NonSaturatingGan(h_size, z_size, learning_rate=lr, use_batchnorm=True),
    "R1 NS-GAN BN γ=10": R1Gan(h_size, z_size, learning_rate=lr, gamma=10.0, use_batchnorm=True),
    "WGAN-GP BN Generator": WGanGP(h_size, z_size, learning_rate=lr, lambd=10.0, use_batchnorm=True),
    "R1 NS-GAN Batch Stats": R1GanBatchStats(h_size, z_size, learning_rate=lr, gamma=10.0),

}
if cuda:
    for name in algorithms.keys():
        algorithms[name] = algorithms[name].cuda()
# dataset = NormalRandomDataset(mean=1.0, stddev=0.5)
# testset = NormalRandomDataset(mean=1.0, stddev=0.5, size=500)
# dataset = MultiNormalDataset(mean1=1.5, std1=0.8, mean2=-1.5, std2=0.5, size=10000)
# testset = MultiNormalDataset(mean1=1.5, std1=0.8, mean2=-1.5, std2=0.5, size=500)
dataset = MultiModeDataset([-1, -0.5, 0, 0.5, 1], size_per_mode=2000, stddev=0.1)
testset = MultiModeDataset([-1, -0.5, 0, 0.5, 1], size_per_mode=100, stddev=0.1)
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
                jsd = evaluation.binned_jsd(testset.data, algorithm, n_bins, bin_range=bin_range)
                js_divergence_values[name].append(jsd)
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
    generated_samples = algorithm.generate_batch(dataset.data.size(0)).view(-1).detach().cpu().numpy()
    real_samples = dataset.data.view(-1).detach().cpu().numpy()
    plt.hist(real_samples, label="Real samples", bins=n_bins, range=bin_range)
    plt.hist(generated_samples, label="Generated samples", bins=n_bins, range=bin_range)
    plt.legend()
    plt.title("Output for %s" % (name,))
    plt.show()

    if plot_d:
        x_values, d_values = algorithm.get_discriminator_values_1d()
        x_values = x_values.view(-1).detach().cpu().numpy()
        d_values = d_values.view(-1).detach().cpu().numpy()
        plt.plot(x_values, d_values, label="D output")
        plt.title("D Output for %s" % (name,))
        plt.show()


try:
    for epoch_number in range(500):
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
