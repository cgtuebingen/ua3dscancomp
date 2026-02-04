import os
import sys

import numpy as np
import pytorch_lightning as pl
import lmdb
import msgpack
import msgpack_numpy as m
import torch

m.patch()
sys.path.append('/home/zakeri/Documents/Codes/MyCodes/Proposal2/ua3dscancomp-gitbub/src/')
# from tqdm import tqdm
from utils.sdf_generatings_fns import setup_cu3d

from utils.partial_view_generation_fns import compute_partial_views, generate_mvp_matrices, create_cuda_raster_context
import utils.shared_dataset_fns as data_fn
#-------------------------------------------------------------------------------------------------------------------------------------
class LMDBOBJAVERSEPARTIALVIEWS(pl.LightningDataModule):
    def __init__(self, mesh_dir: str, lmdb_path: str, res_dir: str, image_resolution: int, resolution: int, device='cuda:0'):

        super(LMDBOBJAVERSEPARTIALVIEWS).__init__()

        try:
            print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
        except KeyError:
            print('CUDA_VISIBLE_DEVICES', 'not set')

        self.mesh_dir = mesh_dir
        self.lmdb_path = lmdb_path
        self.res_dir = res_dir
        self.resolution = resolution

        self.image_resolution = image_resolution
        self.device = device

        self.num_views_to_generate = 20
        self.mvps, self.view_dirs = generate_mvp_matrices(self.num_views_to_generate)

        self.my_lmdb = None
        # print("\n after generate view_dirs class init: num_views: ", num_views, " view_dirs: ", len(self.view_dirs))
        # self.view_selection_mode = view_selection_mode

        self.len: int
        if (os.path.isdir(lmdb_path)):  # if the database exists already:
            with lmdb.open(
                    lmdb_path,
                    max_dbs=2,
                    readonly=True,
                    lock=True,
                    readahead=True,
                    map_size=32 * 1024 * 1024 * 1024 * 1024,
                    max_readers=50,
            ) as my_lmdb:
                with my_lmdb.begin(write=False) as lmdb_txn:  # read it
                    self.mesh_dir = msgpack.unpackb(lmdb_txn.get(b"__mesh_dir__"))
                    self.keys = msgpack.unpackb(lmdb_txn.get(b"__keys__"))  # list of keys
                    self.len = len(self.keys)
        else:  # if it does not exist
            raise RuntimeError("LMDB DOES NOT EXIST!")

    def __len__(self):
        # print("\n dataset class len: num_views: ", self.num_views)
        return self.len

    # def generate_partial_views(self, vertices, faces, res_dir: str, device: str): # pytorch3D
    #     num_views_to_generate = 50
    #     class_obj = GeneratePartialViewFromMesh(torch.from_numpy(vertices).to(device=device), torch.from_numpy(faces).to(device=device), num_views_to_generate, res_dir, device)
    #
    #     dist = [8]
    #     start = time.time()
    #     partial_meshes = class_obj.forward(dist)
    #     end = time.time()
    #     print("\n time for ", num_views_to_generate, " of views to generate, is:", end - start)
    #     return partial_meshes
    # def init_combined_data(self, num_views: int) -> Tuple[float, float, float]:
    #     # priors for 100views
    #     # w_combined = 10.0
    #     # mu_combined = -0.5
    #     # uncertainty_combined = 100.0
    #
    #     # # priors for 20views
    #     # w_combined = 1.0
    #     # mu_combined = -0.5
    #     # uncertainty_combined = 100.0
    #
    #     # # priors for 100views GT
    #     w_combined = 10.0 * (num_views/100)  # weighted based on num_views
    #     mu_combined = -0.03
    #     uncertainty_combined = 100.0
    #
    #     return (mu_combined, uncertainty_combined, w_combined)

    # def add_to_combined_data(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], partial_mesh: Tuple[torch.Tensor, torch.Tensor], resolution: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #
    #     mu_combined, uncertainty_combined, w_combined = data
    #     vertices, faces = partial_mesh
    #
    #     gt_sdf_voxel, gt_dot_product_voxel = self.calculate_sdf_and_dots(vertices, faces, device)
    #     assert (gt_sdf_voxel.shape == gt_dot_product_voxel.shape == (resolution, resolution, resolution))
    #     # plot_sdf(gt_sdf_voxel, i)
    #     gt_sdf_voxel = gt_sdf_voxel.flatten()
    #     gt_dot_product_voxel = gt_dot_product_voxel.flatten()
    #
    #     # plot_sdf(gt_sdf_voxel, i)
    #     gt_sdf_voxel = gt_sdf_voxel.flatten()
    #     gt_dot_product_voxel = gt_dot_product_voxel.flatten()
    #     # calculate uncertainty values here
    #     unc_values = calculate_uncertainty_h8(gt_dot_product_voxel, gt_sdf_voxel).to(device=device)
    #
    #     # combine distributions right here:
    #     mu_combined, uncertainty_combined, w_combined = combine_distribution(gt_sdf_voxel, unc_values, w_combined, mu_combined, uncertainty_combined, device)
    #
    #     if (torch.any(torch.isnan(w_combined)) or torch.any(torch.isinf(w_combined))):
    #         breakpoint()
    #
    #     if (torch.any(torch.isnan(uncertainty_combined)) or torch.any(torch.isinf(uncertainty_combined))):
    #         breakpoint()
    #
    #     return (mu_combined, uncertainty_combined, w_combined)
    # def normalize_combined_data(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    #     mu_combined, uncertainty_combined, w_combined = data
    #
    #     normalized_uncertainty_combined = ((uncertainty_combined / w_combined)).to(device=device)
    #
    #     normalized_mu_combined = (mu_combined).to(device=device)
    #
    #     return (normalized_mu_combined, normalized_uncertainty_combined)
    #
    # def combine_data_all(self, partial_meshes: list, resolution: int, device: str = 'cpu'):
    #
    #     data = self.init_combined_data()
    #
    #     # for i in tqdm(range(0, len(partial_meshes), 1), desc=" gt sdf + dot_product data"):
    #     for i in range(len(partial_meshes)):
    #         data = self.add_to_combined_data(data, partial_meshes[i], resolution, device)
    #
    #     # normalize
    #     return self.normalize_combined_data(data, device)

    # def calculate_sdf_and_dots(self, vertices: torch.Tensor, faces: torch.Tensor, device: str):
    #     if device == 'cpu':
    #         sdf, dots = calculate_sdf_and_dots_cuda(self.cu3d_instance, vertices.cpu().numpy(), faces.cpu().numpy(), device)
    #     else:
    #         sdf, dots = calculate_sdf_and_dots_cuda(self.cu3d_instance, vertices.to(device=device), faces.to(dtype=torch.int32).to(device=device), device)
    #
    #     if (torch.any(torch.isnan(sdf)) or torch.any(torch.isinf(sdf))):
    #         breakpoint()
    #
    #     if (torch.any(torch.isnan(dots)) or torch.any(torch.isinf(dots))):
    #         breakpoint()
    #
    #     return sdf, dots
    #
    # def do_sdf_on_marched_sdf(self, gt_sdf_voxel: np.ndarray) -> dict:
    #     # calculate sdf_latent_code from marched decoded sdf latent_code---------------------------------------------------------------------------
    #     gt_sdf_voxel__copy = np.array(gt_sdf_voxel, copy=True)
    #     gt_sdf_voxel__copy = np.pad(gt_sdf_voxel__copy, pad_width=1, constant_values=-1)   # TODO, Important, checkme
    #     marched_vertices_, marched_triangles_ = mcubes.marching_cubes(gt_sdf_voxel__copy, 0)
    #
    #     # test if they are not empty
    #     if (marched_vertices_.size == 0 or marched_triangles_.size == 0):
    #         return None
    #
    #     assert (gt_sdf_voxel__copy.shape == (128 + 2, 128 + 2, 128 + 2))
    #     # add +0.5 for center
    #     # subtract 1 for padding
    #     # divide by half res ==> extents of 2
    #     # subtract 1 ==> center to 0, bbox [-1, 1]
    #     marched_vertices_ = (marched_vertices_ - 0.5) / 64.0 - 1.0  # assuming the resolution is 128 , bbx of the voxel changes by mcubes, we need to turn it back
    #     # now calculate sdf again------------------------------------------------------------------------------------------------------------------
    #     marched_vertices_ = torch.from_numpy(marched_vertices_.astype(dtype=np.float32))
    #     marched_triangles_ = torch.from_numpy(marched_triangles_.astype(dtype=np.int32))
    #     marched_sdf, marched_dots = self.calculate_sdf_and_dots(marched_vertices_, marched_triangles_, self.device)
    #     # we do not care for the marched_dots
    #     return {"marched_sdf": marched_sdf}
    def setup_cuda_stuff(self):
        # setup cuda stuff
        self.cu3d_instance = setup_cu3d(self.device)
        self.mvps = self.mvps.to(device=self.device)
        self.view_dirs = self.view_dirs.to(device=self.device)
        self.glctx = create_cuda_raster_context(self.device)

    def __getitem__(self, idx: int):
        if self.my_lmdb is None:  # if database object is none
            # print(os.getpid(), 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.is_available())
            print('\n Dataloader', os.getpid(), 'using device', self.device)

            # assert not torch.cuda.is_initialized()
            self.my_lmdb = data_fn.openLMDB(self.lmdb_path)  # create an object and open the database
            self.setup_cuda_stuff()

            # if torch.get_num_threads() < 2:
            #     torch.set_num_threads(8)

        if idx < 0 or idx is None:
            raise "invalid item index"

        if idx > len(self.keys):
            idx = idx % len(self.keys)  # reduce the idx to the len(keys)
        key = self.keys[idx]

        # example = {"mesh_file_name": mesh_file_current, "faces": faces, "vertices": vertices, "bbx": bbx}

        with self.my_lmdb.begin(write=False) as lmdb_txn:  # reading what is written before using the object
            raw_example = msgpack.unpackb(lmdb_txn.get(msgpack.packb(key)))

            mesh_file_name = raw_example["mesh_file_name"]
            mesh_name = raw_example["mesh_name"]
            folder_name = raw_example["folder_name"]
            faces = np.array(raw_example["faces"])
            vertices = np.array(raw_example["vertices"])
            # bbx = np.array(raw_example["bbx"])

            gt_sdf_latent_codes = np.array(raw_example["marched_sdf_latent_codes"], copy=True)
            # gt_uncertainty_voxel = np.array(raw_example["gt_uncertainty_voxel"], copy=True)

        # print("\n dataset class len: getitem1: ", self.num_views)

        #  Now for this item that is extracted from LMDB we need to:
        # 1- generate partial views on the gpu
        # partial_meshes = self.generate_partial_views(vertices, faces, self.res_dir, 'cpu')
        # assert not torch.cuda.is_initialized()

        vertices = torch.from_numpy(vertices).to(device=self.device)
        faces = torch.from_numpy(faces).to(dtype=torch.int32).to(device=self.device)

        # if (torch.any(torch.isnan(vertices)) or torch.any(torch.isinf(vertices))):
        #     breakpoint()
        #
        # if (torch.any(torch.isnan(faces)) or torch.any(torch.isinf(faces))):
        #     breakpoint()

        #  pick a value randomly from the predefined range
        if False:
            chosen_num_views = torch.randperm(self.num_views_to_generate)[:1] + 1  # randperm  produces from 0 to n-1
            num_views = chosen_num_views.item()

        if True:
            chosen_num_views = torch.randperm(4)[:1] + 1  # randperm  produces from 0 to n-1
            num_views = chosen_num_views.item()
        data = data_fn.init_combined_data(num_views)

        # TODO: RANDOM SUBSET OF VIEW_DIRS AND MVPS based on self.num_views
        assert num_views <= len(self.view_dirs)
        # old implementation
        # random_views = torch.randperm(len(self.view_dirs))[0:self.num_views]
        # view_dirs = self.view_dirs[random_views]
        # mvps = self.mvps[random_views]

        # new implementation
        mode = torch.randint(0, 3, (1,)).item()
        # if self.view_selection_mode == 'random':
        if (mode == 2 and num_views > 10):  # it does not make sense to do stratification if we have more than half of total num views
            mode = 0

        if mode == 0:
            random_views = torch.randperm(len(self.view_dirs))[0:num_views]
            # print("\n random_views:", random_views)
            # print("\n random")
            view_dirs = self.view_dirs[random_views]
            # print("\n random_views: ", random_views, "num_views: ", self.num_views)
            mvps = self.mvps[random_views]

        # elif self.view_selection_mode == 'cont':
        elif mode == 1:

            max_ = len(self.view_dirs) - num_views + 1  # upperbound is excluded so we need to add
            # print("\n max_:", max_, "view_dirs: ", len(self.view_dirs), "  num_views: ", self.num_views)
            random_views = torch.randint(0, max_, (1,))
            random_views = torch.arange(num_views) + random_views.item()
            # print("\n continues chunk")

            view_dirs = self.view_dirs[random_views]
            # print("\n random_views: ", random_views, "num_views: ", self.num_views)
            mvps = self.mvps[random_views]

        # elif self.view_selection_mode == 'stratified':
        elif mode == 2:
            chunk_size = (len(self.view_dirs) + num_views - 1) // num_views
            random_views = torch.randint(0, chunk_size, (num_views,)) + torch.arange(0, chunk_size * num_views, chunk_size)
            random_views = random_views % len(self.view_dirs)

            view_dirs = self.view_dirs[random_views]
            mvps = self.mvps[random_views]

        # number of views to rasterize at once
        num_good_views = 0
        raster_batch_size = 4
        # # TODO change me this is Temporary
        # good_partial_meshes = []

        for i in range(0, len(view_dirs), raster_batch_size):
            partial_meshes = compute_partial_views(vertices, faces, self.glctx, mvps[i:i+raster_batch_size], view_dirs[i:i+raster_batch_size], self.image_resolution, self.device)

            # 3- calculate uncertainty values on the gpu
            # combined_sdf, combined_uncertainty = self.combine_data_all(partial_meshes, self.resolution, self.device)
            for partial_mesh in partial_meshes:
                data = data_fn.add_to_combined_data(self.cu3d_instance, data, partial_mesh, self.resolution, self.device)

                num_good_views += 1
                # # TODO change me this is Temporary
                # good_partial_meshes.append(partial_mesh)

            del partial_meshes
        # # TODO change me this is Temporary
        # # march it write here
        # for k in range(len(good_partial_meshes)):
        #     pvertices, pfaces = good_partial_meshes[k]
        #     tri_mesh = trimesh.Trimesh(vertices=pvertices.cpu().numpy(), faces=pfaces.cpu().cpu())
        #     outname = os.path.join(self.res_dir, "_ObjID=" + str(key) + "_k=" + str(k) + "_mode=" + str(mode) + ".obj")
        #     tri_mesh.export(outname)
        #     # print()

        del vertices
        del faces

        if (num_good_views <= 0):
            print("\n Warning: Skipping sampleIDX: ", idx)
            return self[torch.randint(0, len(self), (1, )).item()]

        combined_sdf, combined_uncertainty = data_fn.normalize_combined_data(data, self.device)

        if (torch.any(torch.isnan(combined_sdf)) or torch.any(torch.isinf(combined_sdf))):
            breakpoint()

        if (torch.any(torch.isnan(combined_uncertainty)) or torch.any(torch.isinf(combined_uncertainty))):
            breakpoint()

        # Here, do marching on the combined sdf and then calculate the sdf again from that and then return it into training pipeline for encoding
        if True:
            combined_sdf_reshaped = combined_sdf.reshape([self.resolution, self.resolution, self.resolution])

            marched_dict = data_fn.do_sdf_on_marched_sdf(self.cu3d_instance, combined_sdf_reshaped.cpu().numpy(), self.device)
            if marched_dict is None:  # check if the marched ones is empty. so we skip it.
                print("\n Warning: Skipping sampleIDX: ", idx)
                return self[torch.randint(0, len(self), (1,)).item()]
            else:
                marched_sdf = marched_dict["marched_sdf"]

        # file_name = os.path.join(self.res_dir , "_id-"+ str(idx))
        # write_uncertainty_as_csv(file_name, combined_sdf, combined_uncertainty)
        # 4- download them to cpus and return them here

        marched_sdf = marched_sdf.cpu()
        combined_uncertainty = combined_uncertainty.cpu()

        combined_uncertainty_reshaped = combined_uncertainty.reshape([self.resolution, self.resolution, self.resolution])
        combined_uncertainty_reshaped_normalized = (combined_uncertainty_reshaped / 50) - 1  # he input is [0, 100], -> normalize to [-1, 1]
        # # # TODO temporary , delete me------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # if mode == 0:
        #     name = "random"
        # elif mode == 1:
        #     name = "cont"
        # elif mode == 2:
        #     name = "strat"
        # out_file_name = os.path.join(self.res_dir + "_combined_sdf" "_ObjID=" + str(key) + "_mode=" + str(name) + "_numViews=" + str(num_views) + ".obj")
        # make_mcubes_from_voxels_obj_with_pad(marched_sdf, out_file_name)
        #
        # # marched_sdf = marched_sdf.squeeze(0)
        # # clipping whatever that is too uncertain just for visualization purpose:
        # # uncertainty_med = torch.median(combined_uncertainty_reshaped)
        # # uncertainty_mean = torch.mean(combined_uncertainty_reshaped)
        # plt.hist(torch.flatten(combined_uncertainty_reshaped))
        # plt.xlabel('Uncertainty')
        # clip_out_file_name = os.path.join(self.res_dir + "_clipped_combined_sdf" "_ObjID=" + str(key) + "_mode=" + str(name) + "_numViews=" + str(num_views) + ".png")
        #
        # plt.savefig(clip_out_file_name)
        # plt.close()
        # uncertainty_thresh = 10
        # clipped_marched_sdf = torch.where(combined_uncertainty_reshaped > uncertainty_thresh, -1, marched_sdf)
        # clip_out_file_name = os.path.join(self.res_dir + "_clipped_combined_sdf" "_ObjID=" + str(key) + "_mode=" + str(name) + "_numViews=" + str(num_views) + ".obj")
        # make_mcubes_from_voxels_obj_with_pad(clipped_marched_sdf, clip_out_file_name)
        # print()
        # TODO until here, only for testing-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # gt_out_file_name = os.path.join(self.res_dir + "_gt_sdf" "_ObjID=" + str(key) + "_mode=" + str(mode) + ".obj")
        # tri_mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().cpu())
        # tri_mesh.export(gt_out_file_name)
        # torch.cuda.empty_cache()
        # assert not torch.cuda.is_initialized()

        # marched_sdf_reshaped = marched_sdf.reshape([self.resolution, self.resolution, self.resolution])

        # print("\n dataset class len: getitem2: ", self.num_views)
        return [key, mesh_file_name, mesh_name, folder_name, marched_sdf, combined_uncertainty_reshaped_normalized, gt_sdf_latent_codes]


# def TestLMDB_pytorch_dataloader():
#     mesh_dir = "/graphics/scratch2/datasets/objaverse1.0_processed/"
#     lmdb_path = "/graphics/scratch3/staff/zakeri/LMDBs/filtered_objaverse_joined_lmdb/_val_withLatentCodes__0_1909.mdb"
#     lmdb_path ="/graphics/scratch3/staff/zakeri/LMDBs/filtered_objaverse_joined_lmdb/_train_combined/"
#     lmdb_path = "/ceph/zakeri/LMDB/filtered_objaverse_joined_lmdb_withLatentCodes/test/_test_withLatentCodes__0_5000.mdb"
#     res_root = "/graphics/scratch2/staff/zakeri/Test/"
#     resolution = 128
#
#     image_resolution = 256
#
#     res_dir = os.path.join(res_root, str(image_resolution) + "_mixed4from20" + "/")
#     if os.path.isdir(res_dir):
#         print("\n path exist")
#     else:
#         os.mkdir(res_dir)
#
#     num_views = 4  # TODO this is hardcoded in the code and num_views is used to pass it in training only from now on
#
#     dataset = LMDBOBJAVERSEPARTIALVIEWS(mesh_dir, lmdb_path, res_dir, image_resolution, resolution, 'cuda:1')
#     # dataset.len = 30
#     print("\n dataset len:", len(dataset))
#
#     torch.multiprocessing.set_start_method('spawn')
#
#     # with torch.autograd.profiler.emit_nvtx():
#         # data_loader = DataLoader(dataset, batch_size=1, num_workers=1, persistent_workers=True, shuffle=True)
#         # debug version
#
#     from tqdm import tqdm
#     from torch.utils.data import DataLoader
#
#     data_loader = DataLoader(dataset, batch_size=6, num_workers=1, prefetch_factor=4, shuffle=True)
#     progress = tqdm(iter(data_loader))
#     # from Visualization.m_cube_fns import make_mcubes_from_voxels_obj_with_pad
#
#     for data in progress:
#         key, mesh_file_name, mesh_name, folder_name, marched_sdf, combined_uncertainty_reshaped, gt_sdf_latent_codes = data
#
#         # out_file_name = os.path.join(res_dir + "_combined_sdf_ObjID_ " + str(key) + ".obj")
#         # marched_sdf = marched_sdf.squeeze(0)
#         # make_mcubes_from_voxels_obj_with_pad(marched_sdf, out_file_name)


# if __name__ == "__main__":
#     TestLMDB_pytorch_dataloader()

