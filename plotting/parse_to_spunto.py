from utils import particle_params
import utils
import numpy as np
import json


def main():

    sim_strs = ["medium_rod"]
    run_nums = [0]
    phi_vals = [0.05]

    for sim_string, run_num in zip(sim_strs, run_nums):
        for phi in phi_vals:
            parse(sim_string, phi, run_number=run_num, last_frame=False)


def parse(sim_string, phi, run_number, last_frame=True):

    particle_diam = particle_params[sim_string]["diam"]
    aspect = particle_params[sim_string]["aspect"]
    density_particle = particle_params[sim_string]["density"]
    dirname = utils.get_dirname(
        particle_diam, density_particle, aspect, restart_num=run_number
    )
    sim_params = json.load(open(dirname + f"params/{phi:0.2g}.json"))
    blob_fact = 1.1 if "rod" in sim_string else 1.2
    a_blob = sim_params["a_blob"] * blob_fact
    L = sim_params["L"]

    fname = dirname + "plotting_data/"
    dat = utils.read_binary_file(fname, phi)
    n_blobs = sim_params["n_blobs"]
    n_steps = dat.shape[0]
    blobs_per_body = sim_params["blobs_per_body"]
    n_bodies = n_blobs // blobs_per_body
    color_seed = np.random.randint(0, 10000)
    rng = np.random.default_rng(color_seed)
    colors = rng.uniform(0, 1, size=n_bodies)

    L_shift = np.array([L, L, L])
    out_dir = "./spunto/"
    out_file = out_dir + f"{sim_string}_phi{phi:0.2g}.spunto"
    start_ind = n_steps - 1 if last_frame else 0

    periodize = True  # moves one blob at a time
    periodize_body = False  # moves entire body at once
    skip_frames = 1
    scale_factor = 1.0

    with open(out_file, "w") as f:
        for i in range(start_ind, n_steps, skip_frames):
            if (i - start_ind) > 0:  # splits frames
                f.write("#\n")
            f.write(f"#Lx={L_shift[0]};Ly={L_shift[1]};Lz={L_shift[2]};\n")
            print(i)

            row = dat[i, :]
            row = row.reshape((-1, 3))
            if periodize:
                row = utils.periodize_r_vecs(row, [L, L, L], np.shape(row)[0])
            row -= 0.5 * L_shift
            n_particles = n_blobs // blobs_per_body

            for j in range(n_particles):
                in_body = (
                    row[j * blobs_per_body : (j + 1) * blobs_per_body, :] * scale_factor
                )
                center = np.mean(in_body, axis=0)
                if periodize_body:
                    shift = -L * np.floor(center / L + 0.5)
                    in_body += shift
                color = colors[j]
                for k in range(blobs_per_body):
                    x = in_body[k, 0]
                    y = in_body[k, 1]
                    z = in_body[k, 2]
                    r = a_blob * scale_factor
                    f.write(f"{x} {y} {z} {r} {color}\n")

            f.write("# frame = " + str(i) + "\n")


if __name__ == "__main__":
    main()
