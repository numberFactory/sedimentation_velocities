import numpy as np
import utils
import os
from utils import particle_params


def main():
    # repeated labels are due to high/low temperature runs for some particles
    labels = [
        "small_heavy_sphere",
        "small_light_sphere",
        "big_sphere",
        "short_rod",
        "medium_rod",
        "long_rod",
        "very_long_rod",
        "small_heavy_sphere",
        "small_light_sphere",
        "big_sphere",
    ]

    aspects = [particle_params[l]["aspect"] for l in labels]
    run_numbers = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    phi_vals = [0.01, 0.05, 0.1, 0.15, 0.2]

    densities = [particle_params[l]["density"] for l in labels]
    diams = [particle_params[l]["diam"] for l in labels]

    parse(
        densities=densities,
        aspects=aspects,
        particle_diam=diams,
        phis=phi_vals,
        run_numbers=run_numbers,
        labels=labels,
    )


def parse(densities, aspects, particle_diam, phis, labels, run_numbers=None):

    all_dat = np.zeros((len(phis), len(particle_diam)))
    all_std = np.zeros((len(phis), len(particle_diam)))
    for i, d in enumerate(particle_diam):
        density_fact = densities[i]
        run_number = run_numbers[i] if run_numbers is not None else 0
        aspect = aspects[i]
        dirname = utils.get_dirname(d, density_fact, aspect, restart_num=run_number)

        vel_z, std_v, _ = v_from_files(phis=phis, avg_percent=50, dirname=dirname)
        all_dat[0 : len(vel_z), i] = vel_z
        all_std[0 : len(std_v), i] = std_v

    dat_labels = [f"{label}" + f",std_{label}" for label in labels]
    header_str = "phi, " + ", ".join(dat_labels)
    dat_w_err = np.zeros((len(phis), len(particle_diam) * 2))
    for i in range(len(particle_diam)):
        dat_w_err[:, 2 * i] = all_dat[:, i]
        dat_w_err[:, 2 * i + 1] = all_std[:, i]
    np.savetxt(
        "velocities.csv",
        np.column_stack((phis, dat_w_err)),
        delimiter=",",
        header=header_str,
        comments="",
        fmt="%0.6g",
    )


def v_from_files(phis, avg_percent, dirname):
    avg_v = []
    std_v = []
    files = os.listdir(dirname)
    phis_in_dir = []
    for f in files:
        if f.endswith(".csv") and f.startswith("v_t_"):
            phis_in_dir.append(float(f.split("_t_")[1].split(".csv")[0]))

    phis = [phi for phi in phis if phi in phis_in_dir]

    for phi in phis:
        fname = dirname + f"v_t_{phi:0.2g}.csv"
        dat = np.loadtxt(fname)
        dat = dat.reshape((-1,))
        n_avg = int(len(dat) * avg_percent / 100)
        avg = np.mean(np.abs(dat[-n_avg:]))
        std = np.std(np.abs(dat[-n_avg:]))
        std_v.append(std)
        avg_v.append(avg)
    return np.array(avg_v), np.array(std_v), np.array(phis)


if __name__ == "__main__":
    main()
