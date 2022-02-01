import torch

from gans.abstract_gan import AbstractGan


def binned_jsd(real_data, gan: AbstractGan, n_bins, bin_range=(-4, 4)):
    """
    Approximates the Jensen-Shannon divergence between the real and generated data using n_bins discrete bins
    :param real_data: An array containing samples from the real distribution
    :param gan: The GAN model approximating the real distribution
    :param n_bins: The number of bins used to approximate the JSD.
        More bins != more better unless you also increase the amount of samples
    :param bin_range: The range spanned by the bins (n_bins are spread over this distance).
        Samples outside of this range are counted into the bins on the edges of the range
    :return: An approximation of the Jensen-Shannon divergence between the real and generated samples
    """
    n_samples = real_data.size(0)

    # Generate the same amount of samples as the real data contains
    generated_data = gan.generate_batch(n_samples)
    real_bin_counts = torch.zeros((n_bins,), dtype=torch.float32)
    gen_bin_counts = torch.zeros((n_bins,), dtype=torch.float32)

    bin_step_size = (bin_range[1] - bin_range[0]) / n_bins

    # Clamp all values between the bin ranges
    generated_data = torch.clamp(generated_data, min=bin_range[0], max=bin_range[1] - bin_step_size)
    real_data = torch.clamp(real_data, min=bin_range[0], max=bin_range[1] - bin_step_size)

    # Compute the respective bin index for each value
    generated_data = torch.floor((generated_data - bin_range[0]) / bin_step_size).to(torch.int32).cpu()
    real_data = torch.floor((real_data - bin_range[0]) / bin_step_size).to(torch.int32).cpu()

    for i in range(n_samples):
        gen_bin_counts[generated_data[i].item()] += 1
        real_bin_counts[real_data[i].item()] += 1

    p_real = real_bin_counts / n_samples
    p_gen = gen_bin_counts / n_samples
    return discrete_jensen_shannon_divergence(p_gen, p_real)


def binned_jsd_2d(real_data, gan: AbstractGan, n_bins, bin_range=(-4, 4)):
    """
    Approximates the Jensen-Shannon divergence between the real and generated data using discrete bins
    :param real_data: An array containing samples from the real distribution
    :param gan: The GAN model approximating the real distribution
    :param n_bins_over_1_axis: The number of bins used to approximate the JSD over both axes.
        Ie. when this is set to 4, 4x4 bins will be used to divide up the space
        More bins != more better unless you also increase the amount of samples
    :param bin_range: The range spanned by the bins (n_bins are spread over this distance).
        Samples outside of this range are counted into the bins on the edges of the range
    :return: An approximation of the Jensen-Shannon divergence between the real and generated samples
    """
    n_samples = real_data.size(0)

    # Generate the same amount of samples as the real data contains
    generated_data = gan.generate_batch(n_samples)
    real_bin_counts = torch.zeros((n_bins, n_bins), dtype=torch.float32)
    gen_bin_counts = torch.zeros((n_bins, n_bins), dtype=torch.float32)

    bin_step_size = (bin_range[1] - bin_range[0]) / n_bins

    # Clamp all values between the bin ranges
    generated_data = torch.clamp(generated_data, min=bin_range[0], max=bin_range[1] - bin_step_size)
    real_data = torch.clamp(real_data, min=bin_range[0], max=bin_range[1] - bin_step_size)

    # Compute the respective bin index for each value
    generated_data = torch.floor((generated_data - bin_range[0]) / bin_step_size).to(torch.int32).cpu()
    real_data = torch.floor((real_data - bin_range[0]) / bin_step_size).to(torch.int32).cpu()

    for i in range(n_samples):
        gen_bin_counts[generated_data[i, 0].item(), generated_data[i, 1].item()] += 1
        real_bin_counts[real_data[i, 0].item(), real_data[i, 1].item()] += 1

    p_real = real_bin_counts.view(-1) / n_samples
    p_gen = gen_bin_counts.view(-1) / n_samples
    return discrete_jensen_shannon_divergence(p_gen, p_real)


def binned_jsd_old(real_data, gan: AbstractGan, n_bins, bin_range=(-4, 4)):
    """
    Approximates the Jensen-Shannon divergence between the real and generated data using n_bins discrete bins
    :param real_data: An array containing samples from the real distribution
    :param gan: The GAN model approximating the real distribution
    :param n_bins: The number of bins used to approximate the JSD.
        More bins != more better unless you also increase the amount of samples
    :param bin_range: The range spanned by the bins (n_bins are spread over this distance).
        Samples outside of this range are counted into the bins on the edges of the range
    :return: An approximation of the Jensen-Shannon divergence between the real and generated samples
    """
    n_samples = real_data.size(0)

    # Generate the same amount of samples as the real data contains
    generated_data = gan.generate_batch(n_samples)
    real_bin_counts = torch.zeros((n_bins,), dtype=torch.float32)
    gen_bin_counts = torch.zeros((n_bins,), dtype=torch.float32)

    bin_step_size = (bin_range[1] - bin_range[0]) / n_bins

    for bin_i in range(n_bins):
        lower_bound = bin_range[0] + bin_step_size * bin_i
        higher_bound = lower_bound + bin_step_size

        if bin_i == 0:
            # Left edge, count everything below upper bound (also samples outside of bin_range)
            real_n = torch.count_nonzero(real_data < higher_bound)
            gen_n = torch.count_nonzero(generated_data < higher_bound)
        elif bin_i == n_bins - 1:
            # Right edge, count everything above lower bound (also samples outside of bin_range)
            real_n = torch.count_nonzero(real_data >= lower_bound)
            gen_n = torch.count_nonzero(generated_data >= lower_bound)
        else:
            # Count samples in bin
            real_n = torch.count_nonzero((real_data >= lower_bound) & (real_data < higher_bound))
            gen_n = torch.count_nonzero((generated_data >= lower_bound) & (generated_data < higher_bound))

        real_bin_counts[bin_i] = real_n
        gen_bin_counts[bin_i] = gen_n

    p_real = real_bin_counts / n_samples
    p_gen = gen_bin_counts / n_samples
    return discrete_jensen_shannon_divergence(p_gen, p_real)


def binned_jsd_2d_old(real_data, gan: AbstractGan, n_bins_over_1_axis, bin_range=(-4, 4)):
    """
    Approximates the Jensen-Shannon divergence between the real and generated data using discrete bins
    :param real_data: An array containing samples from the real distribution
    :param gan: The GAN model approximating the real distribution
    :param n_bins_over_1_axis: The number of bins used to approximate the JSD over both axes.
        Ie. when this is set to 4, 4x4 bins will be used to divide up the space
        More bins != more better unless you also increase the amount of samples
    :param bin_range: The range spanned by the bins (n_bins are spread over this distance).
        Samples outside of this range are counted into the bins on the edges of the range
    :return: An approximation of the Jensen-Shannon divergence between the real and generated samples
    """
    n_samples = real_data.size(0)

    # Generate the same amount of samples as the real data contains
    generated_data = gan.generate_batch(n_samples)
    real_bin_counts = torch.zeros((n_bins_over_1_axis * n_bins_over_1_axis,), dtype=torch.float32)
    gen_bin_counts = torch.zeros((n_bins_over_1_axis * n_bins_over_1_axis,), dtype=torch.float32)

    bin_step_size = (bin_range[1] - bin_range[0]) / n_bins_over_1_axis

    for bin_x in range(n_bins_over_1_axis):
        lower_bound_x = bin_range[0] + bin_step_size * bin_x
        higher_bound_x = lower_bound_x + bin_step_size

        if bin_x == 0:
            # Left edge, count everything below upper bound (also samples outside of bin_range)
            real_in_x = real_data < higher_bound_x
            gen_in_x = generated_data < higher_bound_x
        elif bin_x == n_bins_over_1_axis - 1:
            # Right edge, count everything above lower bound (also samples outside of bin_range)
            real_in_x = real_data >= lower_bound_x
            gen_in_x = generated_data >= lower_bound_x
        else:
            # Count samples in bin
            real_in_x = (real_data >= lower_bound_x) & (real_data < higher_bound_x)
            gen_in_x = (generated_data >= lower_bound_x) & (generated_data < higher_bound_x)

        for bin_y in range(n_bins_over_1_axis):
            lower_bound_y = bin_range[0] + bin_step_size * bin_y
            higher_bound_y = lower_bound_y + bin_step_size

            if bin_y == 0:
                # Left edge, count everything below upper bound (also samples outside of bin_range)
                real_in_y = real_data < higher_bound_y
                gen_in_y = generated_data < higher_bound_y
            elif bin_y == n_bins_over_1_axis - 1:
                # Right edge, count everything above lower bound (also samples outside of bin_range)
                real_in_y = real_data >= lower_bound_y
                gen_in_y = generated_data >= lower_bound_y
            else:
                # Count samples in bin
                real_in_y = (real_data >= lower_bound_y) & (real_data < higher_bound_y)
                gen_in_y = (generated_data >= lower_bound_y) & (generated_data < higher_bound_y)

            real_n = torch.count_nonzero(real_in_x & real_in_y)
            gen_n = torch.count_nonzero(gen_in_x & gen_in_y)

            real_bin_counts[bin_x + n_bins_over_1_axis * bin_y] = real_n
            gen_bin_counts[bin_x + n_bins_over_1_axis * bin_y] = gen_n

    p_real = real_bin_counts / n_samples
    p_gen = gen_bin_counts / n_samples
    return discrete_jensen_shannon_divergence(p_gen, p_real)


def discrete_kullback_leibler_divergence(p1, p2):
    """
    Computes the discrete Kullback-Leibler Divergence between 2 1-dimensional arrays that
        both represent a probability distribution where every value in the array equals the probability of a sampled
        value being in that bin. Formally: D_KL( p1 || p2 ).
        This method may deliver incorrect results when p1 is 0 where p2 is not
    :param p1: The first probability distribution
    :param p2: The second probability distribution
    :return: D_KL( p1 || p2 )
    """
    eps = 1e-10
    summands = p1 * torch.log((p1 + eps) / (p2 + eps))

    return torch.sum(summands)


def discrete_jensen_shannon_divergence(p1, p2):
    """
    Computes the discrete Jensen-Shannon Divergence between 2 1-dimensional arrays that
        both represent a probability distribution where every value in the array equals the probability of a sampled
        value being in that bin. Formally: JSD( p1 || p2 ).
        This method may deliver incorrect results when either p1 or p2 contains a 0 in one place where the other does not
    :param p1: The first probability distribution
    :param p2: The second probability distribution
    :return: JSD( p1 || p2 )
    """
    m = 0.5 * (p1 + p2)
    return 0.5 * (discrete_kullback_leibler_divergence(p1, m) + discrete_kullback_leibler_divergence(p2, m))
