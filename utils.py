import numpy as np
from numba import njit, prange
import scipy.spatial as spatial
import os
from enum import Enum
import json

particle_params = {
    "big_sphere": {"diam": 0.9, "density": 2.2, "aspect": 1},
    "small_heavy_sphere": {"diam": 0.3, "density": 2.2, "aspect": 1},
    "small_light_sphere": {"diam": 0.3, "density": 1.05, "aspect": 1},
    "short_rod": {"diam": 0.532, "density": 1.9, "aspect": 3},
    "medium_rod": {"diam": 0.445, "density": 1.9, "aspect": 7},
    "long_rod": {"diam": 0.384, "density": 1.9, "aspect": 11},
    "very_long_rod": {"diam": 0.273, "density": 1.9, "aspect": 20},
}


class particleType(Enum):
    SPHERE = 1
    ROD = 2


def get_dirname(diam_sphere, density_fact, aspect, restart_num=None):
    if restart_num is not None:
        run_number = restart_num
    else:
        run_number = 0
    dirname = f"output/aspect_{aspect:d}/d_{diam_sphere:0.1g}/rho_{density_fact:0.2g}/run_{run_number:d}/"
    if restart_num is not None:
        return dirname

    while os.path.exists(dirname):
        run_number += 1
        dirname = f"output/aspect_{aspect:d}/d_{diam_sphere:0.1g}/rho_{density_fact:0.2g}/run_{run_number:d}/"
    os.makedirs(dirname)
    restart_dir = dirname + "restarts/"
    os.makedirs(restart_dir)
    plot_dir = dirname + "plotting_data/"
    os.makedirs(plot_dir)
    params_dir = dirname + "params/"
    os.makedirs(params_dir)
    return dirname


def write_row_binary(out_dir, row, N, phi, out_dtype="float32") -> None:
    pos_file = out_dir + f"blobs_{phi:0.2g}.bin"
    meta_file = out_dir + f"binary_metadata_{phi:0.2g}.json"

    row = np.array(row, dtype=out_dtype)
    if not os.path.exists(meta_file):
        metadata = {
            "row_size": row.size,
            "N": N,
            "n_rows": 1,  # account for row about to be written
            "dtype": out_dtype,
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)
    else:
        with open(meta_file, "r") as f:
            metadata = json.load(f)
            metadata["n_rows"] += 1
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)

    with open(pos_file, "ab") as f:
        row.tofile(f)


def read_binary_file(dir, phi) -> np.ndarray:
    pos_file = dir + f"blobs_{phi:0.2g}.bin"
    meta_file = dir + f"binary_metadata_{phi:0.2g}.json"

    with open(meta_file, "r") as f:
        metadata = json.load(f)

    n_rows = metadata["n_rows"]
    row_size = metadata["row_size"]
    dtype = np.dtype(metadata["dtype"])

    data = np.fromfile(pos_file, dtype=dtype)
    data = data.reshape((n_rows, row_size))
    return data


def load_cfg(file_name, particle_type=particleType.SPHERE) -> tuple[dict, np.ndarray]:
    with open(file_name, "r") as f:
        _ = f.readline()
        params = f.readline().strip().split(",")
        if particle_type == particleType.SPHERE:
            sep = float(params[0].split(" ")[1])
            N = int(params[1])
            rg = float(params[2])
            rh = int(params[3])
            params = {"sep": sep, "N": N, "Rg": rg, "Rh": rh}
        elif particle_type == particleType.ROD:
            sep = float(params[0].split(" ")[1])
            N = int(params[1])
            diameter = float(params[2])
            length = float(params[3])
            params = {"sep": sep, "N": N, "diameter": diameter, "length": length}
        else:
            raise ValueError("Unknown particle type")

        cfg = np.loadtxt(f, delimiter=" ")

    return params, cfg


@njit(parallel=True, fastmath=True)
def blob_blob_sterics(
    r_vectors,
    blobs_per_body,
    L,
    a,
    repulsion_strength,
    debye_length,
    list_of_neighbors,
    offsets,
    neighbor_cut,
):
    N = r_vectors.size // 3
    force = np.zeros((N, 3))

    for i in prange(N):
        body_ind_i = i // blobs_per_body

        for kk in range(offsets[i + 1] - offsets[i]):
            j = list_of_neighbors[offsets[i] + kk]
            body_ind_j = j // blobs_per_body

            if body_ind_i == body_ind_j:  # skip self-interaction
                continue

            dr = np.zeros(3)
            for k in range(3):
                dr[k] = r_vectors[j, k] - r_vectors[i, k]
                if L[k] > 0:
                    dr[k] -= (
                        int(dr[k] / L[k] + 0.5 * (int(dr[k] > 0) - int(dr[k] < 0)))
                        * L[k]
                    )

            r_norm = np.sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2])

            offset = 2.0 * a
            if r_norm > neighbor_cut:
                continue

            temp_r = max(r_norm, 1.0e-12)
            inv_r_norm = 1 / temp_r
            if r_norm > offset:
                prefactor = (
                    -(repulsion_strength / debye_length)
                    * np.exp(-(r_norm - offset) / debye_length)
                    * inv_r_norm
                )
            else:
                prefactor = -(repulsion_strength / debye_length) * inv_r_norm

            force[i] += prefactor * dr

    return force


def build_neighbor_list(r_vectors, L, r_cut):
    r_vectors = np.reshape(r_vectors, (-1, 3))
    r_vectors = periodize_r_vecs(r_vectors, L, np.shape(r_vectors)[0])

    r_tree = spatial.KDTree(
        r_vectors, boxsize=L + 0.01, balanced_tree=False, compact_nodes=False
    )

    pairs = r_tree.query_ball_point(r_vectors, r_cut, return_sorted=False, workers=1)

    offsets = np.cumsum([0] + [len(p) for p in pairs], dtype=int)
    list_of_neighbors = np.fromiter(
        (item for sublist in pairs for item in sublist), dtype=int
    )
    return offsets, list_of_neighbors


@njit(parallel=True, fastmath=True)
def periodize_r_vecs(r_vecs_np, L, Nb):
    r_vecs = np.copy(r_vecs_np)
    for k in prange(Nb):
        for i in range(3):
            if L[i] > 0:
                while r_vecs[k, i] < 0:
                    r_vecs[k, i] += L[i]
                while r_vecs[k, i] > L[i]:
                    r_vecs[k, i] -= L[i]
    return r_vecs
