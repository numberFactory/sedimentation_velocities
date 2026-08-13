import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def main():

    fname = "./velocities.csv"
    dat = pd.read_csv(fname)
    col_names = dat.columns.tolist()
    colnames = col_names[1::2]
    vel_z = dat.iloc[:, 1::2].to_numpy()
    std_v = dat.iloc[:, 2::2].to_numpy()
    phi = dat.iloc[:, 0].to_numpy()

    labels = [
        "sphere, d=0.3$\\mu$m, $\\rho=2.2$, T=20 $^{\\circ}$C",
        "sphere, d=0.3$\\mu$m, $\\rho=1.05$, T=20 $^{\\circ}$C",
        "sphere, d=0.9$\\mu$m, $\\rho=2.2$, T=20 $^{\\circ}$C",
        "rod, aspect=3",
        "rod, aspect=7",
        "rod, aspect=11",
        "rod, aspect=20",
        "sphere, d=0.3$\\mu$m, $\\rho=2.2$, T=90 $^{\\circ}$C",
        "sphere, d=0.3$\\mu$m, $\\rho=1.05$, T=90 $^{\\circ}$C",
        "sphere, d=0.9$\\mu$m, $\\rho=2.2$, T=90 $^{\\circ}$C",
    ]

    fig, (ax_rod, ax_sphere) = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True, constrained_layout=True
    )

    cmap_rod = plt.get_cmap("viridis")
    cmap_sphere = plt.get_cmap("inferno")

    sphere_handles = []
    rod_handles = []

    high_temp_ind = 7

    for i in range(vel_z.shape[1]):

        is_rod = "rod" in colnames[i]
        is_low_temp_sphere = (not is_rod) and i < 3

        if is_rod:
            color_ind = (i - 3) / 4 + 0.15
            color = cmap_rod(color_ind)
            marker = "s"
            axes_to_plot = [ax_rod]
            handle_lists = [rod_handles]
        elif is_low_temp_sphere:
            color_ind = i / 5 + 0.45
            color = cmap_sphere(color_ind)
            marker = "o"
            axes_to_plot = [ax_sphere, ax_rod]
            handle_lists = [sphere_handles, []]
        else:
            color_ind = i if i < high_temp_ind else i - 7
            color_ind = color_ind / 5 + 0.45
            color = cmap_sphere(color_ind)
            marker = "o" if i < high_temp_ind else "^"
            axes_to_plot = [ax_sphere]
            handle_lists = [sphere_handles]

        linestyle = "--" if "highT" in colnames[i] else "-"

        mask = vel_z[:, i] > 0

        for ax, handle_list in zip(axes_to_plot, handle_lists):
            h = ax.errorbar(
                phi[mask],
                vel_z[mask, i],
                yerr=std_v[mask, i],
                label=labels[i],
                capsize=5,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=8,
                lw=2,
            )
            handle_list.append(h)

    for ax in (ax_sphere, ax_rod):
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 5)
        ax.set_xticks([0.01, 0.05, 0.1, 0.15, 0.2])
        ax.set_xlabel("Volume fraction $\\phi$")

    ax_rod.set_ylabel("Sedimentation velocity ($\\mu$m/s)")

    legend_handles = [Line2D([], [], linestyle="none", marker="")] + rod_handles
    legend_labels = [r"$\rho = 1.9$"] + [handle.get_label() for handle in rod_handles]
    ax_rod.legend(
        handles=legend_handles,
        labels=legend_labels,
        fontsize=12,
        loc="upper right",
    )

    plt.savefig("sed_velocities.svg", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
