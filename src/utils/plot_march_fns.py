import torch
from typing import Any

from training.subvolume_devision import (
    collect_sub_voxels_to_voxel_with_batch,
)
from utils.m_cube_fns import make_mcubes_from_voxels_obj_for_pad

def decode_data_for_vis(data_vis: torch.Tensor, dict_arguments_of_variables: dict, fdecoder):
    number_of_sub_voxels = dict_arguments_of_variables["number_of_sub_voxels"]
    target_resolution = dict_arguments_of_variables["target_resolution"]
    batch_size = dict_arguments_of_variables["batch_size"]

    with torch.no_grad():
        decoded_data_vis = fdecoder(data_vis).to(device=fdecoder.device)  # [1, 64, 512, 2, 2, 2] -> [64, 1, 32, 32, 32]
        decoded_data_vis_reshaped = decoded_data_vis.reshape([batch_size, number_of_sub_voxels, target_resolution, target_resolution, target_resolution])

    return decoded_data_vis_reshaped

def generate_any_data_for_plotting(dict_arguments_for_vis: dict, dict_arguments_of_variables: dict, fdecoder) -> dict:
    resolution = dict_arguments_of_variables["resolution"]
    batch_size = dict_arguments_of_variables["batch_size"]
    keys = [key for key in dict_arguments_for_vis.keys()]
    collected_data_vis: dict = {}

    for i in range(len(keys)):
        current_key = keys[i]
        data_vis_current = dict_arguments_for_vis.get(current_key).detach()

        decoded_data_vis_current_reshaped = decode_data_for_vis(data_vis_current, dict_arguments_of_variables, fdecoder)
        collected_decoded_data_vis_current_reshaped = collect_sub_voxels_to_voxel_with_batch(decoded_data_vis_current_reshaped, resolution)
        assert collected_decoded_data_vis_current_reshaped.shape == (batch_size, resolution, resolution, resolution)
        collected_data_vis[current_key] = collected_decoded_data_vis_current_reshaped

    return collected_data_vis

def collect_any_generated_data_for_plotting(data_dict_for_vis: dict, batch_idx: int) -> dict[str, Any]:
    keys = [key for key in data_dict_for_vis.keys()]
    return_dict = dict()
    for i in range(len(keys)):
        current_key = keys[i]
        collected_data_current = data_dict_for_vis.get(current_key).detach()
        collected_data_current_array = collected_data_current[batch_idx].squeeze().detach().to(torch.float32).cpu().numpy()
        return_dict[current_key] = collected_data_current_array
    return return_dict


def march_any_results_every_n_epoch(dict_of_items: dict, selected_index: int, current_epoch: int, global_step: int, marching_cube_result_dir: str) -> None:
    if (global_step // 5000) == 0:
        keys = [key for key in dict_of_items.keys()]
        return_dict = dict()
        for i in range(len(keys)):
            current_key = keys[i]
            collected_data_current = dict_of_items.get(current_key)
            current_step = global_step
            current_name = str(current_key)
            make_mcubes_from_voxels_obj_for_pad(collected_data_current, selected_index, current_name, marching_cube_result_dir)

