import time
import numpy as np
import scipy.spatial as spatial
from Rigid import RigidBody
from libMobility import PSE
import pyamg
from scipy.sparse.linalg import LinearOperator
import utils
import os
import json

from utils import particle_params


def run():

    # when true, this loads a config from the equilibrium_configs directory and makes a directory indicated by
    # the restart_num. to use restarts, set this to false and use restart_num to indicate the config to load.
    use_equilibrium_start = False

    ### low temp configs for sphers and rods

    sim_strs = [
        "big_sphere",
        "small_heavy_sphere",
        "small_light_sphere",
        "short_rod",
        "medium_rod",
        "long_rod",
        "very_long_rod",
    ]
    restart_nums = [0, 0, 0, 0, 0, 0, 0]
    phi_vals = [
        [0.01, 0.05, 0.1, 0.15, 0.2],
        [0.01, 0.05, 0.1, 0.15, 0.2],
        [0.01, 0.05, 0.1, 0.15, 0.2],
        [0.01, 0.05, 0.1, 0.15, 0.2],
        [0.01, 0.05, 0.1, 0.15],
        [0.01, 0.05, 0.1],
        [0.01, 0.05],
    ]
    dt_mults = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    # n_tau is the number of sedimentation timescales to run for. a simulation should be run til converged,
    # which may require increasing this number or using restarts.
    n_tau_facts = [5, 5, 5, 5, 5, 5, 5]
    high_temp = False

    ### high temp configs for spheres

    # sim_strs = ["small_heavy_sphere", "small_light_sphere", "big_sphere"]
    # restart_nums = [1, 1, 1]
    # phi_vals = [
    #     [0.01, 0.05, 0.1, 0.15, 0.2],
    #     [0.01, 0.05, 0.1, 0.15, 0.2],
    #     [0.01, 0.05, 0.1, 0.15, 0.2],
    # ]
    # dt_mults = [0.1, 0.1, 0.2]
    # n_tau_facts = [0.3, 0.3, 3.0]
    # high_temp = True

    for sim_str, restart_num, phi_val, dt_mult, n_tau_fact in zip(
        sim_strs, restart_nums, phi_vals, dt_mults, n_tau_facts
    ):
        main(
            sim_str,
            phi_val,
            dt_mult,
            n_tau_fact,
            restart_num=restart_num,
            use_equilibrium_start=use_equilibrium_start,
            high_temp=high_temp,
        )


def main(
    sim_string,
    phi_vals,
    dt_mult,
    n_tau_fact,
    restart_num=None,
    high_temp=False,
    use_equilibrium_start=False,
):
    particle_diam = particle_params[sim_string]["diam"]
    aspect = particle_params[sim_string]["aspect"]
    particle_type = utils.particleType.SPHERE if aspect == 1 else utils.particleType.ROD
    density_particle = particle_params[sim_string]["density"]

    L_fact = 50.0
    save_blob_data = False
    dirname = utils.get_dirname(
        particle_diam, density_particle, aspect, restart_num=restart_num
    )

    domain_size_radii = 0.3
    L = L_fact * 0.5 * domain_size_radii

    for phi in phi_vals:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            os.makedirs(dirname + "/params/")
            os.makedirs(dirname + "/restarts/")

        print("RUNNING CASE:", sim_string, "phi =", phi)
        get_velocity(
            particle_diam,
            density_particle,
            phi,
            dt_mult,
            n_tau_fact,
            particle_type=particle_type,
            high_temp=high_temp,
            aspect=aspect,
            L=L,
            use_equilibrium_start=use_equilibrium_start,
            dirname=dirname,
            restart_number=restart_num,
            save_plot_data=save_blob_data,
        )


def get_velocity(
    particle_diam,
    density_particle,
    phi,
    dt_mult,
    n_tau_fact,
    particle_type,
    aspect,
    dirname,
    L,
    high_temp=False,
    use_equilibrium_start=False,
    restart_number=None,
    save_plot_data=True,
):
    dtype = np.float64
    R_particle = 0.5 * particle_diam
    n_plot = 100
    n_save_blob_data = 25

    if high_temp and particle_type == utils.particleType.ROD:
        raise ValueError("High temp option only valid for spheres")
    fluid_density = 1.0 if not high_temp else 0.965
    eta = 1e-3 if not high_temp else 0.3142 * 1e-3  # Pa.s

    if particle_type == utils.particleType.SPHERE:
        ico_int = 162
        struct_file = f"structures/shell_N_{ico_int}.csv"
    elif particle_type == utils.particleType.ROD:
        struct_file = f"structures/rod_aspect_{aspect}.csv"
    else:
        raise ValueError("Unknown particle type")

    cfg_params, cfg = utils.load_cfg(struct_file, particle_type=particle_type)
    # spheres are unit radius, rods are already correct size
    if particle_type == utils.particleType.SPHERE:
        cfg *= R_particle
        a_blob = 0.5 * spatial.distance.pdist(cfg).min()
    else:
        a_blob = 0.5 * cfg_params["sep"]

    X0, Q0, n_particles = load_starting_positions(
        dirname,
        phi,
        aspect,
        particle_type,
        high_temp=high_temp,
        equilibrium_start=use_equilibrium_start,
        restart_number=restart_number,
    )

    g = 9.81e6  # um/s^2
    density = (density_particle - fluid_density) * 1e-9  # mg/um^3
    if particle_type == utils.particleType.SPHERE:
        particle_volume = (4.0 / 3.0) * np.pi * R_particle**3
        mass_sphere = density * particle_volume
        mg = mass_sphere * g
        m0 = 1 / (6 * np.pi * eta * R_particle)
    elif particle_type == utils.particleType.ROD:
        L_rod = cfg_params["length"]
        R_rod = 0.5 * cfg_params["diameter"]

        L_tmp = L_rod - 2 * R_rod  # length of cylinder part of rod
        particle_volume = np.pi * R_rod**2 * L_tmp + (4.0 / 3.0) * np.pi * R_rod**3
        mass_rod = density * particle_volume
        mg = mass_rod * g
        m0 = np.log(aspect) / (4 * np.pi * eta * L_rod)
    else:
        raise ValueError("Unknown particle type")

    v_g = -mg * m0

    len_scale = particle_volume ** (1 / 3)
    repulsion_fact = 4.0
    print("Using repulsion fact:", repulsion_fact)

    repulsion_strength = repulsion_fact * mg * len_scale
    debye_length = 0.1 * a_blob
    delta_cut = 2 * a_blob
    r_cut = 2 * a_blob + 4 * debye_length + delta_cut
    # tau_sterics = (6 * np.pi * eta * a_blob**2 * debye_length) / repulsion_strength

    tau_sedimentation = a_blob / abs(v_g)
    sedimentation_tau_fact = dt_mult
    t_final_mult = n_tau_fact
    print("using sedimentation tau fact:", sedimentation_tau_fact)
    dt = sedimentation_tau_fact * tau_sedimentation

    tau_final = (t_final_mult * L) / abs(v_g)
    n_t = int(tau_final / dt)
    print("Total sedimentation time (s):", tau_final)
    print("Number of time steps:", n_t)
    print("Number of particles, phi", n_particles, phi)
    print("phi calc", n_particles * particle_volume / L**3)
    blobs_per_body = cfg.shape[0]
    n_blobs = n_particles * blobs_per_body

    params = {
        "a_sphere": R_particle,
        "blobs_per_body": blobs_per_body,
        "n_particles": n_particles,
        "n_blobs": n_blobs,
        "a_blob": a_blob,
        "eta": eta,
        "fluid_density": fluid_density,
        "particle_density": density_particle,
        "phi": phi,
        "L": L,
        "eta": eta,
        "density": density,
        "sedimentation_tau_fact": sedimentation_tau_fact,
        "sedimentation_tau": tau_sedimentation,
        "dt": dt,
        "v_g": v_g,
        "repulsion_fact": repulsion_fact,
        "repulsion_strength": repulsion_strength,
        "debye_length": debye_length,
        "mg": mg,
        "particle_type": particle_type.name,
        "n_t_max": n_t,
    }
    with open(dirname + f"params/{phi:0.2g}.json", "w") as f:
        json.dump(params, f, indent=4)

    rb, solver = create_solvers(
        cfg=cfg,
        X0=X0,
        Q0=Q0,
        n_blobs=n_blobs,
        L=L,
        a_blob=a_blob,
        eta=eta,
        dt=dt,
        n_particles=n_particles,
    )

    def apply_A(x):
        out = np.zeros_like(x)
        blob_size = 3 * n_blobs
        lam = x[0:blob_size]
        U = x[blob_size : blob_size + 6 * n_particles]

        mf = solver.Mdot(forces=lam)[0].flatten()
        kt_U = rb.K_dot(U).flatten()
        out[0:blob_size] = mf - kt_U

        out[blob_size:] = -rb.KT_dot(lam).flatten()

        return out.astype(dtype)

    def apply_PC(x):
        out = rb.apply_PC(x)
        return out.astype(dtype)

    blob_pos = np.array(rb.get_blob_positions())
    offsets, nlist = utils.build_neighbor_list(
        blob_pos, np.array([L, L, L]), r_cut=r_cut
    )
    pos_x0 = blob_pos.copy()

    # note: F includes torque components, but they are zero here
    F_g = np.tile([0.0, 0.0, -mg, 0.0, 0.0, 0.0], (n_particles, 1)).astype(dtype)
    if restart_number is not None and not use_equilibrium_start:
        v_t_prev = np.loadtxt(dirname + f"v_t_{phi:0.2g}.csv", delimiter=",")
        n_t_prev = v_t_prev.shape[0]
        v_t = np.zeros(n_t + n_t_prev)
        v_t[0:n_t_prev] = v_t_prev
        start_t = n_t_prev
        n_t += n_t_prev
    else:
        start_t = 0
        v_t = np.zeros(n_t)

    n_rebuilds = 0
    start = time.time()
    for t in range(start_t, n_t):
        blob_pos = np.array(rb.get_blob_positions())
        blob_disp = np.linalg.norm(
            blob_pos.reshape((-1, 3)) - pos_x0.reshape((-1, 3)), axis=1
        )
        if np.any(blob_disp > delta_cut):
            n_rebuilds += 1
            offsets, nlist = utils.build_neighbor_list(
                blob_pos, np.array([L, L, L]), r_cut=r_cut
            )
            pos_x0 = blob_pos.copy()

        f_sterics = utils.blob_blob_sterics(
            blob_pos.reshape((-1, 3)),
            blobs_per_body,
            np.array([L, L, L]),
            a_blob,
            repulsion_strength,
            debye_length,
            list_of_neighbors=nlist,
            offsets=offsets,
            neighbor_cut=r_cut,
        )
        F = F_g.flatten() + rb.KT_dot(f_sterics.flatten()).flatten()

        RHS = np.zeros(3 * n_blobs + 6 * n_particles, dtype=dtype)
        RHS[3 * n_blobs : 3 * n_blobs + 6 * n_particles] = -F.flatten()
        RHS_norm = np.linalg.norm(RHS).astype(dtype)

        solver.setPositions(blob_pos)

        size = 3 * n_blobs + 6 * n_particles
        A = LinearOperator(shape=(size, size), matvec=apply_A, dtype=dtype)  # type: ignore
        PC = LinearOperator(shape=(size, size), matvec=apply_PC, dtype=dtype)  # type: ignore
        res_list = []
        sol, _ = pyamg.krylov.gmres(
            A,
            (RHS / RHS_norm),
            x0=None,
            tol=1e-3,
            M=PC,
            maxiter=min(300, size),
            restrt=None,
            residuals=res_list,
        )
        sol *= RHS_norm
        U = sol[3 * n_blobs : 3 * n_blobs + 6 * n_particles]
        rb.evolve_rigid_bodies(U.flatten())
        v_t[t] = np.mean(U[2::6])

        if save_plot_data and t % n_save_blob_data == 0:
            fname = dirname + f"plotting_data/"
            row = rb.get_blob_positions().flatten()
            utils.write_row_binary(fname, row, n_blobs, phi)

        if t % 50 == 0:
            end = time.time()
            print("time:", t, " average vel z:", v_t[t])
            print(
                "Time for 50 steps (s):",
                end - start,
                "gmres iters:",
                len(res_list),
                "nlist rebuilds:",
                n_rebuilds,
            )
            n_rebuilds = 0
            start = time.time()

        if t % n_plot == 0:
            save_stuff(phi, dirname, rb, v_t, t - 1)

    print("Simulation complete, saving final data...")
    save_stuff(phi, dirname, rb, v_t, n_t - 1)


def load_starting_positions(
    dirname,
    phi,
    aspect,
    particle_type,
    high_temp=None,
    equilibrium_start=False,
    restart_number=None,
):
    if equilibrium_start and phi > 0.0:
        non_eq_file = dirname + f"restarts/last_config_{phi:0.2g}.csv"
        if os.path.exists(non_eq_file):
            raise ValueError(
                "Non-equilibrium restart file found. Aborting instead of overwriting."
            )
        dirname = dirname.split("output/")[1]
        eq_file = "equilibrium_configs/" + dirname.split("run_")[0]
        if particle_type == utils.particleType.SPHERE:
            if high_temp is None:
                raise ValueError("Must specify high_temp for sphere equilibrium start")
            if high_temp:
                eq_file += "high_t/"
            else:
                eq_file += "low_t/"
        eq_file += "restarts/"
        eq_file += f"last_config_{phi:0.2g}.csv"
        dat = np.loadtxt(eq_file, delimiter=",")
        n_particles = dat.shape[0]
        X0 = dat[:, 0:3]
        Q0 = dat[:, 3:7]
        return X0, Q0, n_particles

    if restart_number is not None and phi > 0.0:
        rst_file = dirname + f"restarts/last_config_{phi:0.2g}.csv"
        dat = np.loadtxt(rst_file, delimiter=",")
        n_particles = dat.shape[0]
        X0 = dat[:, 0:3]
        Q0 = dat[:, 3:7]
        np.savetxt(
            dirname + f"restarts/initial_config_{phi:0.2g}.csv", dat, delimiter=","
        )
        return X0, Q0, n_particles

    if phi == 0.0:
        X0 = np.array([0.0, 0.0, 0.0])
        z = 0.5 * np.pi * np.array([0.0, 1.0, 0.0])
        Q0 = spatial.transform.Rotation.from_rotvec(z).as_quat().reshape((1, 4))
        n_particles = 1
    elif aspect == 1:
        pack_file = f"./{dirname}configs/sphere_pack_{phi:0.3g}.txt"
        X0 = np.loadtxt(pack_file, delimiter=" ", skiprows=1)
        n_particles = X0.shape[0]
        Q0 = np.tile([1.0, 0.0, 0.0, 0.0], (n_particles, 1))
    elif aspect > 1:
        pack_file = (
            f"./rod_packings/s1/rod_packings_aspect{aspect}_phi{int(phi*100)}.csv"
        )
        dat = np.loadtxt(pack_file, delimiter=",")
        X0 = dat[:, 0:3] / 1000.0  # convert from nm
        Q0 = dat[:, 3:7]
        print("max/min rod coords", X0.max(axis=0), X0.min(axis=0))
        n_particles = X0.shape[0]
    else:
        raise ValueError("Unknown configuration for particles")

    return X0, Q0, n_particles


def save_stuff(phi, dirname, rb, v_t, t):

    fname = dirname + f"restarts/last_config_{phi:0.2g}.csv"
    X_tmp, Q_tmp = rb.get_config()
    X_tmp = X_tmp.reshape((-1, 3))
    Q_tmp = Q_tmp.reshape((-1, 4))
    dat = np.hstack((X_tmp, Q_tmp)).reshape((-1, 7))
    np.savetxt(fname, dat, delimiter=",")

    fname = dirname + f"v_t_{phi:0.2g}.csv"
    np.savetxt(fname, v_t[0:t], delimiter=",")


def create_solvers(cfg, X0, Q0, n_blobs, L, a_blob, eta, dt, n_particles):
    rb = RigidBody(cfg, X0, Q0, a_blob, eta, dt)

    psi_fact = 2.0 if n_particles == 1 else 1.0
    psi = psi_fact * n_blobs ** (1 / 3) / L
    solver = PSE("periodic", "periodic", "periodic")
    solver.setParameters(Lx=L, Ly=L, Lz=L, psi=psi)
    solver.initialize(viscosity=eta, hydrodynamicRadius=a_blob)

    return rb, solver


if __name__ == "__main__":
    run()
