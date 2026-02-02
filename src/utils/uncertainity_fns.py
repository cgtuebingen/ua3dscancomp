import os
import torch
from typing import Tuple
from tqdm import tqdm
import numpy as np
def calculate_uncertainty_h7(dot_product_value, dist):
    # print("\n min :", torch.min(dot_product_value), ", max: ", torch.max(dot_product_value))
    unc_val = ((dist * dist * 100.0) * (2 - torch.pow(dot_product_value, 100)))
    return unc_val

def calculate_uncertainty_h8(dot_product_value, dist):
    # print("\n min :", torch.min(dot_product_value), ", max: ", torch.max(dot_product_value))
    unc_val = ((dist * dist * 100.0) * (2 - torch.pow(dot_product_value, 21)) + 50.0 * torch.square(1.0-torch.abs(dot_product_value)))

    # unc_val = ((dist * dist * 100.0) * (1.5 - 0.5*torch.pow(dot_product_value, 9))).to(device=device)
    # unc_val = ((dist * dist * 100.0) * (2 - torch.pow(torch.relu(dot_product_value), 20))).to(device=device)
    return unc_val

def calculate_uncertainty_h8_grid_search(dot_product_value, dist, weight_dist: float = 100.0, wight_dot_product: float = 50.0):
    # print("\n min :", torch.min(dot_product_value), ", max: ", torch.max(dot_product_value))
    unc_val = ((dist * dist * weight_dist) * (2 - torch.pow(dot_product_value, 21)) + wight_dot_product * torch.square(1.0-torch.abs(dot_product_value)))

    # unc_val = ((dist * dist * 100.0) * (1.5 - 0.5*torch.pow(dot_product_value, 9))).to(device=device)
    # unc_val = ((dist * dist * 100.0) * (2 - torch.pow(torch.relu(dot_product_value), 20))).to(device=device)
    return unc_val
def combine_distribution(sdf: torch.Tensor, uncertainty: torch.Tensor, w_combined: torch.float32, mu_combined: torch.float32, uncertainty_combined: torch.float32):
    sigma_i = uncertainty
    mu_i = sdf

    w_i = (1 / (sigma_i + 1e-3))
    w_combined = (w_combined + w_i)

    diff_old = (mu_i - mu_combined)
    mu_combined = (mu_combined + (w_i / w_combined) * diff_old)
    diff_new = (mu_i - mu_combined)

    uncertainty_combined = (uncertainty_combined + w_i * diff_old * diff_new)
    return (mu_combined, uncertainty_combined, w_combined)

def combine_distributions_with_incremental_mean(sdf_all: list[torch.Tensor], uncertainty_all: list[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # mu here is sdf
    # sigma is the variance of distribution which is uncertainty values
    assert (len(sdf_all) == len(uncertainty_all))

    w_combined = 10.0
    mu_combined = -0.5
    uncertainty_combined = 100.0

    for i in range(len(sdf_all)):
        sigma_i = uncertainty_all[i]
        mu_i = sdf_all[i]

        w_i = (1/(sigma_i + 1e-3))
        w_combined = (w_combined + w_i)

        diff_old = (mu_i - mu_combined)
        mu_combined = (mu_combined + (w_i/w_combined) * diff_old)
        diff_new = (mu_i - mu_combined)

        uncertainty_combined = (uncertainty_combined + w_i * diff_old * diff_new)

    normalized_uncertainty_combined = (uncertainty_combined/w_combined)
    normalized_mu_combined = (mu_combined)

    # print("\n mean mu_combined", torch.mean(mu_combined))
    # print("\n mean w_combined", torch.mean(w_combined))
    # print("\n mean normalized_mu_combined", torch.mean(mu_combined))
    # print("\n mean normalized_uncertainty_combined", torch.mean(normalized_uncertainty_combined))

    return (normalized_mu_combined, normalized_uncertainty_combined)


def generate_coordinates(resolution: int = 128, value_range: int = 1):
    # generate coordinates:
    x_ = np.linspace(-value_range, value_range, resolution, False, dtype=np.float32)
    y_ = np.linspace(-value_range, value_range, resolution, False, dtype=np.float32)
    z_ = np.linspace(-value_range, value_range, resolution, False, dtype=np.float32)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    voxels = np.stack((x, y, z), axis=3) + (value_range - -value_range) / (2 * resolution)
    points = voxels.reshape([-1, 3])
    return points, voxels
def write_uncertainty_as_csv(file_name: str, gt_sdf_voxel: torch.Tensor,  unc_values: torch.Tensor):
    points, voxels = generate_coordinates()

    string = ''
    for p in range(points.shape[0]):
        point_current = points[p, :]
        string += str(point_current[0].item()) + ', ' + str(point_current[1].item()) + ', ' + str(point_current[2].item()) + ', ' + str(gt_sdf_voxel[p].item()) + ', ' + str(unc_values[p].item()) + '\n'

    out_name = file_name + "_unc_combined_all_h8_prior.csv"
    with open(out_name, 'w') as f:
        f.write(string)