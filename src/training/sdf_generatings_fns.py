import sys
sys.path.append('..')

# sys.path.append('/home/zakeri/Documents/Codes/MyCodes/Proposal2/Partial3DScanCompletion/pycu3d/build/')
import torch
import numpy as np
#from pycu3d.build import cu3d
import cu3d
# from pycu3d.test.evaluate_all import EVALALLMETRICS
#from Partial3DScan.Developement.uncertainty_processing.uncertainity_fns import write_uncertainty_as_csv
def generate_sdf_cuda(cu3d_instance: cu3d.CU3D, py3d_mesh, brute_force: bool = False, device: str = 'cuda:0'):

    if brute_force:
        sdfs, dots = cu3d_instance.compute_sdf_grid_brute_force(py3d_mesh, np.array([-1, -1, -1]), np.array([1, 1, 1]), 128)
    else:
        sdfs, dots = cu3d_instance.compute_sdf_grid_with_bvh(py3d_mesh, np.array([-1, -1, -1]), np.array([1, 1, 1]), 128)

    if device == 'cpu':
        sdfs = -torch.from_numpy(sdfs.cpu()).transpose(0, 2)  # inside is positive
        dots = torch.from_numpy(dots.cpu()).transpose(0, 2)
    else:
        sdfs = -sdfs.cuda().transpose(0, 2)  # inside is positive
        dots = dots.cuda().transpose(0, 2)

    # print("\n sdf: ", sdfs.shape, ", dot product:", dots.shape)

    if (torch.any(torch.isnan(sdfs)) or torch.any(torch.isinf(sdfs))):
        breakpoint()
        # print("\n sdfs: ", sdfs)

    if (torch.any(torch.isnan(dots)) or torch.any(torch.isinf(dots))):
        # print("\ndots: ", dots)
        breakpoint()

    return {"gt_sdf_voxel": sdfs, "gt_dot_product_voxel": dots}
def calculate_sdf_and_dots_cuda(cu3d_instance, vertices: torch.Tensor, faces: torch.Tensor, device):
    # make a pycuda mesh object
    vertices = cu3d.GPUPointData(vertices)
    faces = cu3d.GPUTriangleData(faces)
    py3d_mesh = cu3d.GPUMesh(vertices, faces)

    sdf_dict = generate_sdf_cuda(cu3d_instance, py3d_mesh, False, device)

    sdf = sdf_dict["gt_sdf_voxel"]
    dots = sdf_dict["gt_dot_product_voxel"]

    # py3d_mesh.save_obj('/graphics/scratch2/datasets/objaverse1.0_processed/objaverse_100/res_dir/debug/mesh.obj')
    # write_uncertainty_as_csv('/graphics/scratch2/datasets/objaverse1.0_processed/objaverse_100/res_dir/debug/sdf.csv', sdf.cpu().flatten(), dots.cpu().flatten())
    return (sdf, dots)

def setup_cu3d(device: str) -> cu3d.CU3D:
    device_id = int(str(device).split(':')[1])
    cu3d.set_gpu_id(device_id)
    cu3d_instance = cu3d.CU3D()

    return cu3d_instance
