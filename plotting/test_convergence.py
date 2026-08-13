from utils import particle_params
import utils
import numpy as np
import matplotlib.pyplot as plt


def main():

    sim_string = "big_sphere"
    restart_num = 0
    phi = 0.01
    particle_diam = particle_params[sim_string]["diam"]
    aspect = particle_params[sim_string]["aspect"]
    density_particle = particle_params[sim_string]["density"]

    dirname = utils.get_dirname(
        particle_diam, density_particle, aspect, restart_num=restart_num
    )
    fname = dirname + f"v_t_{phi:0.2g}.csv"
    dat = np.loadtxt(fname).reshape(-1)

    plt.figure()
    plt.plot(dat)
    plt.title("Mean velocity")
    plt.savefig("mean_vel.png")
    plt.close()

    n_blocks = 10
    f_mean, b_mean, f_std, b_std = forward_backward_convergence(dat, n_blocks)
    all_mean = f_mean[-1]
    all_std = f_std[-1]

    plt.figure(figsize=(8, 6))
    x = np.arange(1, n_blocks + 1)
    plt.errorbar(
        x,
        f_mean,
        yerr=f_std,
        label="Forward Mean",
        fmt="o-",
        color="blue",
    )
    plt.errorbar(
        x,
        b_mean,
        yerr=b_std,
        label="Backward Mean",
        fmt="o-",
        color="orange",
    )
    all_mean = np.mean(dat)
    all_std = np.std(dat)
    plt.axhline(all_mean, color="black", linestyle="--", label="Overall Mean")
    plt.axhspan(
        all_mean - 0.25 * all_std,
        all_mean + 0.25 * all_std,
        color="gray",
        alpha=0.3,
        label="Overall Std Dev",
    )
    plt.axvline(n_blocks // 2, color="black", linestyle=":", label="Halfway Point")
    plt.legend(loc="lower left")
    plt.title(f"{sim_string}, phi={phi}")
    plt.savefig("velocity_convergence.png", dpi=300)
    plt.close()

    thresh = 0.25 * all_std
    f_converged = np.abs(np.array(f_mean) - all_mean) < thresh
    b_converged = np.abs(np.array(b_mean) - all_mean) < thresh
    both_converged = f_converged & b_converged
    if not np.all(both_converged[n_blocks // 2 :]):
        print(f"sim {sim_string} phi {phi} NOT converged")
        print(f_converged)
        print(b_converged)
    else:
        print(
            "sim {sim_string} phi {phi} converged".format(
                sim_string=sim_string, phi=phi
            )
        )


def forward_backward_convergence(dat, num_blocks):
    N = len(dat)
    block_size = N // num_blocks

    forward_means = []
    backward_means = []
    forward_std = []
    backward_std = []

    for i in range(1, num_blocks + 1):
        forward_block = dat[: i * block_size]
        backward_block = dat[N - i * block_size :]

        forward_means.append(np.mean(forward_block))
        backward_means.append(np.mean(backward_block))

        forward_std.append(np.std(forward_block))
        backward_std.append(np.std(backward_block))

    return (forward_means, backward_means, forward_std, backward_std)


if __name__ == "__main__":
    main()
