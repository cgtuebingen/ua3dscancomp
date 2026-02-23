import sys
import numpy as np
import lmdb

import msgpack_numpy as m

import torch
from typing import Tuple

m.patch()
sys.path.append("..")

from utils.sdf_generatings_fns import calculate_sdf_and_dots_cuda
from utils.uncertainity_fns import (
    calculate_uncertainty_h8,
    combine_distribution,
    calculate_uncertainty_h8_grid_search,
)
import mcubes
from utils.partial_view_generation_fns import generate_mvp_matrices


def openLMDB(path: str):
    my_lmdb = lmdb.open(
        path,
        # max_dbs=2,
        readonly=True,  # we just want to read it
        lock=False,  # reading!!
        readahead=True,
        map_size=32 * 1024 * 1024 * 1024,
        # max_readers=10000,
    )
    my_lmdb.open_db()
    return my_lmdb


def init_combined_data(num_views: int) -> Tuple[float, float, float]:

    # # priors for 100views GT
    w_combined = 10.0 * (num_views / 100)  # weighted based on num_views
    mu_combined = -0.03
    weighted_uncertainty_combined = 10.0 * w_combined  # 50->20 views to start with

    return (mu_combined, weighted_uncertainty_combined, w_combined)


def add_to_combined_data(
    cu3d_instance,
    data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    partial_mesh: Tuple[torch.Tensor, torch.Tensor],
    resolution: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu_combined, uncertainty_combined, w_combined = data
    vertices, faces = partial_mesh

    gt_sdf_voxel, gt_dot_product_voxel = calculate_sdf_and_dots(
        cu3d_instance, vertices, faces, device
    )
    assert (
        gt_sdf_voxel.shape
        == gt_dot_product_voxel.shape
        == (resolution, resolution, resolution)
    )
    gt_sdf_voxel = gt_sdf_voxel.flatten()
    gt_dot_product_voxel = gt_dot_product_voxel.flatten()

    # calculate uncertainty values here
    unc_values = calculate_uncertainty_h8(gt_dot_product_voxel, gt_sdf_voxel)

    # combine distributions right here:
    mu_combined, uncertainty_combined, w_combined = combine_distribution(
        gt_sdf_voxel, unc_values, w_combined, mu_combined, uncertainty_combined
    )

    return (mu_combined, uncertainty_combined, w_combined)


def add_to_combined_data_grid_search(
    cu3d_instance,
    data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    partial_mesh: Tuple[torch.Tensor, torch.Tensor],
    resolution: int,
    device: str,
    weight_dist: float = 100.0,
    wight_dot_product: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    mu_combined, uncertainty_combined, w_combined = data
    vertices, faces = partial_mesh

    gt_sdf_voxel, gt_dot_product_voxel = calculate_sdf_and_dots(
        cu3d_instance, vertices, faces, device
    )
    assert (
        gt_sdf_voxel.shape
        == gt_dot_product_voxel.shape
        == (resolution, resolution, resolution)
    )
    gt_sdf_voxel = gt_sdf_voxel.flatten()
    gt_dot_product_voxel = gt_dot_product_voxel.flatten()

    # calculate uncertainty values here
    unc_values = calculate_uncertainty_h8_grid_search(
        gt_dot_product_voxel, gt_sdf_voxel, weight_dist, wight_dot_product
    )

    # combine distributions right here:
    mu_combined, uncertainty_combined, w_combined = combine_distribution(
        gt_sdf_voxel, unc_values, w_combined, mu_combined, uncertainty_combined
    )

    return (mu_combined, uncertainty_combined, w_combined)


def normalize_combined_data(
    data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    mu_combined, weighted_uncertainty_combined, w_combined = data

    normalized_uncertainty_combined = ((weighted_uncertainty_combined / w_combined)).to(
        device=device
    )

    normalized_mu_combined = (mu_combined).to(device=device)

    return (normalized_mu_combined, normalized_uncertainty_combined)


def combine_data_all(partial_meshes: list, resolution: int, device: str = "cpu"):
    data = init_combined_data()

    for i in range(len(partial_meshes)):
        data = add_to_combined_data(data, partial_meshes[i], resolution, device)

    # normalize
    return normalize_combined_data(data, device)


def calculate_sdf_and_dots(
    cu3d_instance, vertices: torch.Tensor, faces: torch.Tensor, device: str
):
    if device == "cpu":
        sdf, dots = calculate_sdf_and_dots_cuda(
            cu3d_instance, vertices.cpu().numpy(), faces.cpu().numpy(), device
        )
    else:
        sdf, dots = calculate_sdf_and_dots_cuda(
            cu3d_instance,
            vertices.to(device=device),
            faces.to(dtype=torch.int32).to(device=device),
            device,
        )

    return sdf, dots


def do_sdf_on_marched_sdf(cu3d_instance, gt_sdf_voxel: np.ndarray, device: str) -> dict:
    # calculate sdf_latent_code from marched decoded sdf latent_code---------------------------------------------------------------------------
    # gt_sdf_voxel__copy = np.array(gt_sdf_voxel, copy=True)
    gt_sdf_voxel__copy = np.pad(gt_sdf_voxel, pad_width=1, constant_values=-1)
    marched_vertices_, marched_triangles_ = mcubes.marching_cubes(gt_sdf_voxel__copy, 0)

    # test if they are not empty
    if marched_vertices_.size == 0 or marched_triangles_.size == 0:
        raise RuntimeError("empty mesh given")

    assert gt_sdf_voxel__copy.shape == (128 + 2, 128 + 2, 128 + 2)
    # add +0.5 for center
    # subtract 1 for padding
    # divide by half res ==> extents of 2
    # subtract 1 ==> center to 0, bbox [-1, 1]
    marched_vertices_ = (
        marched_vertices_ - 0.5
    ) / 64.0 - 1.0  # assuming the resolution is 128 , bbx of the voxel changes by mcubes, we need to turn it back
    # now calculate sdf again------------------------------------------------------------------------------------------------------------------
    marched_vertices_ = torch.from_numpy(marched_vertices_.astype(dtype=np.float32))
    marched_triangles_ = torch.from_numpy(marched_triangles_.astype(dtype=np.int32))
    marched_sdf, marched_dots = calculate_sdf_and_dots(
        cu3d_instance, marched_vertices_, marched_triangles_, device
    )
    # we do not care for the marched_dots
    return {
        "marched_sdf": marched_sdf,
        "marched_vertices": marched_vertices_,
        "marched_faces": marched_triangles_,
    }


def do_sdf_on_marched_sdf_cuda(
    cu3d_instance, gt_sdf_voxel: np.ndarray, device: str
) -> dict:
    # calculate sdf_latent_code from marched decoded sdf latent_code---------------------------------------------------------------------------
    gt_sdf_voxel__copy = torch.from_numpy(gt_sdf_voxel).to(device=device)
    gt_sdf_voxel__copy = torch.nn.functional.pad(
        gt_sdf_voxel__copy, pad=(1, 1, 1, 1, 1, 1), value=-1
    )  # TODO, Important, checkme

    marched_vertices_, marched_triangles_ = mcubes.marching_cubes(gt_sdf_voxel__copy, 0)
    marched_triangles_ = torch.stack(
        [marched_triangles_[:, 0], marched_triangles_[:, 2], marched_triangles_[:, 1]],
        dim=1,
    )

    # test if they are not empty
    if marched_vertices_.size == 0 or marched_triangles_.size == 0:
        return None

    assert gt_sdf_voxel__copy.shape == (128 + 2, 128 + 2, 128 + 2)
    # add +0.5 for center
    # subtract 1 for padding
    # divide by half res ==> extents of 2
    # subtract 1 ==> center to 0, bbox [-1, 1]
    marched_vertices_ = (
        marched_vertices_ - 0.5
    ) / 64.0 - 1.0  # assuming the resolution is 128 , bbx of the voxel changes by mcubes, we need to turn it back
    # now calculate sdf again------------------------------------------------------------------------------------------------------------------
    marched_sdf, marched_dots = calculate_sdf_and_dots(
        cu3d_instance, marched_vertices_, marched_triangles_, device
    )
    # we do not care for the marched_dots
    return {"marched_sdf": marched_sdf}


def pick_camera_views_per_object(
    num_views: int = 7, num_test_split: int = 5000, num_views_to_generate: int = 20
):
    mvps, view_dirs = generate_mvp_matrices(num_views_to_generate)
    view_dirs_selected_all = []
    mvps_selected_all = []

    chunk_size = (len(view_dirs) + num_views - 1) // num_views
    random_views = torch.randint(
        0, chunk_size, (num_test_split, num_views)
    ) + torch.arange(0, chunk_size * num_views, chunk_size).unsqueeze(0)
    backup_views = torch.randint(
        chunk_size * (num_views - 1), len(view_dirs), (num_test_split, num_views)
    )
    random_views[random_views >= len(view_dirs)] = backup_views[
        random_views >= len(view_dirs)
    ]

    shuffled_views = []

    for i in range(num_test_split):
        permutation = torch.randperm(num_views)

        shuffled_views.append(random_views[i, permutation])

    shuffled_views = torch.vstack(shuffled_views)

    mvps_selected_all = mvps[shuffled_views]
    view_dirs_selected_all = view_dirs[shuffled_views]
    # torch.save({'mvps': mvps, 'view_dirs': view_dirs, 'shuffled_views': shuffled_views}, "./test_views/views.pkl") # for objaverse
    # return (mvps, view_dirs, shuffled_views)


# if __name__ == "__main__":
#     pick_camera_views_per_object(7, 1500, 20)
