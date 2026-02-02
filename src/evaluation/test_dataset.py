import os
import sys
sys.path.append('/home/zakeri/Documents/Codes/MyCodes/Proposal2/ua3dscancomp-gitbub/src/')
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pytorch_lightning as pl
import lmdb
import msgpack
import msgpack_numpy as m
import torch
m.patch()
sys.path.append('/home/zakeri/Documents/Codes/MyCodes/Proposal2/Partial3DScanCompletion/')
from utils.sdf_generatings_fns import setup_cu3d

from utils.partial_view_generation_fns import compute_partial_views, generate_mvp_matrices, create_cuda_raster_context
import mcubes
import trimesh
from utils.m_cube_fns import make_mcubes_from_voxels_obj_with_pad
import utils.shared_dataset_fns as data_fn
import utils.shared_lmd_fns as lmdb_fn
#-------------------------------------------------------------------------------------------------------------------------------------
class TESTLMDBOBJAVERSEPARTIALVIEWS(pl.LightningDataModule):
    def __init__(self, mesh_dir: str, lmdb_path: str, res_dir: str, image_resolution: int, num_views: int, resolution: int, device='cuda:0'):

        super(TESTLMDBOBJAVERSEPARTIALVIEWS).__init__()

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

        # TODO : I changed here because we do that hardcoded
        # self.test_mode = test_mode
        # self.num_views_to_generate = 20
        # self.mvps, self.view_dirs = generate_mvp_matrices(self.num_views_to_generate)
        views_dict = torch.load("/graphics/scratch3/staff/zakeri/LMDBs/filtered_objaverse_joined_lmdb/test_views/views.pkl")
        self.mvps = views_dict['mvps']
        self.view_dirs = views_dict['view_dirs']
        self.shuffled_views = views_dict['shuffled_views']

        self.my_lmdb = None
        self.num_views = num_views

        self.len: int
        if (os.path.isdir(lmdb_path)):  # if the database exists already:
            my_lmdb = lmdb_fn.openLMDB(lmdb_path)
            with my_lmdb.begin(write=False) as lmdb_txn:  # read it
                self.keys = msgpack.unpackb(lmdb_txn.get(b"__keys__"))  # list of keys
                self.len = len(self.keys)
        else:  # if it does not exist
            raise RuntimeError("LMDB DOES NOT EXIST!")

    def __len__(self):
        return self.len #* len(self.num_views)

    def setup_cuda_stuff(self):
        # setup cuda stuff
        self.cu3d_instance = setup_cu3d(self.device)
        self.mvps = self.mvps.to(device=self.device)
        self.view_dirs = self.view_dirs.to(device=self.device)
        self.glctx = create_cuda_raster_context(self.device)

    def __getitem__(self, idx: int):
        if self.my_lmdb is None:  # if database object is none
            print('\n Dataloader', os.getpid(), 'using device', self.device)
            self.my_lmdb = lmdb_fn.openLMDB(self.lmdb_path)  # create an object and open the database
            self.setup_cuda_stuff()

        if idx < 0 or idx is None:
            raise "invalid item index"

        num_views = self.num_views[idx % len(self.num_views)]
        idx = idx // len(self.num_views)

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

        #  Now for this item that is extracted from LMDB we need to:
        # 1- generate partial views on the gpu
        # if True:
        #     flat_mesh = trimesh.load("/graphics/scratch3/staff/zakeri/Paper_3DPartialScanComletion/Figures/Uncertatiny/flatten_heart.2obj.obj", force='mesh', merge_norm=True, merge_text=True)
        #     vertices = np.array(flat_mesh.vertices, dtype=np.float32)
        #     faces = np.array(flat_mesh.faces)

        vertices = torch.from_numpy(vertices).to(device=self.device)
        faces = torch.from_numpy(faces).to(dtype=torch.int32).to(device=self.device)
        # for num_views in self.num_view_list:

        data = data_fn.init_combined_data(num_views)

        # pick views from shuffled views----------------------------------------------------------------------------------------------------------------------------------------------------------------
        assert num_views <= len(self.view_dirs)
        random_views = self.shuffled_views[key, 0: num_views]
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
        # # # march it write here
        if False:
            for k in range(len(good_partial_meshes)):
                pvertices, pfaces = good_partial_meshes[k]
                tri_mesh = trimesh.Trimesh(vertices=pvertices.cpu().numpy(), faces=pfaces.cpu().cpu())
                folder_id_path= os.path.join("/graphics/scratch3/staff/zakeri/ObjaverseEval/selected_for_video/", str(key))
                if not os.path.isdir(folder_id_path):
                    os.mkdir(folder_id_path)
                partial_path  = os.path.join(folder_id_path, "Partial" + "_num_views-" + str(self.num_views))
                if not os.path.isdir(partial_path):
                    os.mkdir(partial_path)
                outname = os.path.join(partial_path, "_ObjID=" + str(key) + "_p=" + str(k) + ".obj")
                tri_mesh.export(outname)
                print()
        # TODO , you might not need me any more, can be commented
        # del vertices
        # del faces

        if (num_good_views <= 0):
            print("\n Warning: Skipping sampleIDX: ", idx)
            return self[torch.randint(0, len(self), (1, )).item()]

        combined_sdf, combined_uncertainty = data_fn.normalize_combined_data(data, self.device)

        if (torch.any(torch.isnan(combined_sdf)) or torch.any(torch.isinf(combined_sdf))):
            breakpoint()

        if (torch.any(torch.isnan(combined_uncertainty)) or torch.any(torch.isinf(combined_uncertainty))):
            breakpoint()

        # Here, do marching on the combined sdf and then calculate the sdf again from that and then return it into training pipeline for encoding
        # if True:   # TODO: this MUST be True always!
        combined_sdf_reshaped = combined_sdf.reshape([self.resolution, self.resolution, self.resolution])

        marched_dict = data_fn.do_sdf_on_marched_sdf(self.cu3d_instance, combined_sdf_reshaped.cpu().numpy(), self.device)
        if marched_dict is None:  # check if the marched ones is empty. so we skip it.
            print("\n Warning: Skipping sampleIDX: ", idx)
            return self[torch.randint(0, len(self), (1,)).item()]
        else:
            marched_sdf = marched_dict["marched_sdf"]

        # # TODO temporary,
        # if False:
        #     from Partial3DScan.Developement.uncertainty_processing.uncertainity_fns import write_uncertainty_as_csv
        #     file_name = os.path.join("/graphics/scratch3/staff/zakeri/Paper_3DPartialScanComletion/Figures/Uncertatiny/" , "_id-" + str(key))
        #     write_uncertainty_as_csv(file_name, combined_sdf, combined_uncertainty)
        #     print()

        # 4- download them to cpus and return them here
        marched_sdf = marched_sdf.cpu()
        combined_uncertainty = combined_uncertainty.cpu()

        # gt_out_file_name = os.path.join(self.res_dir + "_gt_sdf" "_ObjID=" + str(key) + "_mode=" + str(mode) + ".obj")
        # tri_mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().cpu())
        # tri_mesh.export(gt_out_file_name)

        # marched_sdf_reshaped = marched_sdf.reshape([self.resolution, self.resolution, self.resolution])
        combined_uncertainty_reshaped = combined_uncertainty.reshape([self.resolution, self.resolution, self.resolution])
        combined_uncertainty_reshaped_normalized = (combined_uncertainty_reshaped / 50) - 1  # he input is [0, 100], -> normalize to [-1, 1]

        if False:
            common_obj_dir = "/graphics/scratch3/staff/zakeri/ObjaverseEval/common_obj_dir/"
            clipped_combined_sdf_voxel = marched_sdf - torch.clip(combined_uncertainty_reshaped - 8, min=0.0) * 0.003
            input_dir = os.path.join(common_obj_dir, "Inputs", "Input" + "_num_views-" + str(self.num_views))
            if not os.path.isdir(input_dir):
                os.mkdir(input_dir)
            input_export_file_name = os.path.join(input_dir, "Input" + "_" + folder_name + "_" + mesh_name.rsplit('_')[0] + "_ObjID=" + str(key) + ".ply")
            if not os.path.isfile(input_export_file_name):
                make_mcubes_from_voxels_ply_with_pad(clipped_combined_sdf_voxel.cpu().numpy(), input_export_file_name)

        # torch.save(combined_uncertainty_reshaped, "/graphics/scratch3/staff/zakeri/Paper_3DPartialScanComletion/Figures/Uncertatiny/flat_combined_uncertainty.pkl")
        # torch.save(marched_sdf, "/graphics/scratch3/staff/zakeri/Paper_3DPartialScanComletion/Figures/Uncertatiny/flat_combined_sdf.pkl")

        # # TODO temporary, delete me
        # if False:
        #     from Partial3DScan.Developement.evaluation.eval_fns import clip_input
        #     out_file_name = os.path.join("/graphics/scratch3/staff/zakeri/Paper_3DPartialScanComletion/Figures/Uncertatiny/" + "_combined_sdf" "_ObjID=" + str(key) + ".obj")
        #     marched_sdf = marched_sdf.squeeze(0)
        #     make_mcubes_from_voxels_obj_with_pad(marched_sdf, out_file_name)
        #
        #     clipped_out_file_name = os.path.join("/graphics/scratch3/staff/zakeri/Paper_3DPartialScanComletion/Figures/Uncertatiny/" + "clipped_combined_sdf" "_ObjID=" + str(key) + ".obj")
        #     clipped_combined_sdf_voxel = clip_input(marched_sdf, combined_uncertainty_reshaped, uncertainty_thresh=20)
        #     make_mcubes_from_voxels_obj_with_pad(clipped_combined_sdf_voxel, clipped_out_file_name)

        return [key, mesh_file_name, mesh_name, folder_name, marched_sdf, combined_uncertainty_reshaped_normalized, vertices, faces, gt_sdf_latent_codes, num_views]

#
# def TestLMDB_pytorch_dataloader():
#     mesh_path = "/graphics/scratch2/datasets/objaverse1.0_processed/"
#     test_lmdb_path = "/graphics/scratch3/staff/zakeri/LMDBs/filtered_objaverse_joined_lmdb/_test_withLatentCodes__0_5000.mdb"
#
#     res_root = "/graphics/scratch3/staff/zakeri/Paper_3DPartialScanComletion/Figures/Uncertatiny/"
#     resolution = 128
#
#     image_resolution = 256
#
#     # res_dir = os.path.join(res_root, str(image_resolution) + "_mixed4from20" + "/")
#     # if os.path.isdir(res_dir):
#     #     print("\n path exist")
#     # else:
#     #     os.mkdir(res_dir)
#     views_list = [4, 3, 2, 1]
#     selected_objects = [3339, 2849, 3701, 2236, 2055, 1336, 1872]
#     selected_objects = [4723]
#
#     for i in range(len(views_list)):
#
#         num_views_for_test = views_list[i]  # TODO this is hardcoded in the code and num_views is used to pass it in training only from now on
#
#         # test_mode = 2
#         test_dataset = TESTLMDBOBJAVERSEPARTIALVIEWS(mesh_path, test_lmdb_path, res_root, image_resolution, num_views_for_test, resolution, device='cuda:0')
#         print("\n test_dataset len:", len(test_dataset))
#
#         # test_dataset.len = 50
#         print("\n test_dataset len:", len(test_dataset))
#         for j in range(0, len(test_dataset), 1):
#             if j in selected_objects:
#                 sample = test_dataset[j]
#                 pass
        # torch.multiprocessing.set_start_method('spawn')

    # with torch.autograd.profiler.emit_nvtx():
        # data_loader = DataLoader(dataset, batch_size=1, num_workers=1, persistent_workers=True, shuffle=True)
        # debug version


    # for i in range(0, len(test_dataset), 1):
    #     if i == 49:
    #         key, mesh_file_name, mesh_name, folder_name, marched_sdf, combined_uncertainty_reshaped, gt_sdf_latent_codes = test_dataset[i]
    # from tqdm import tqdm
    # from torch.utils.data import DataLoader
    #
    # data_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, persistent_workers=True)
    # progress = tqdm(iter(data_loader))
    # # from Visualization.m_cube_fns import make_mcubes_from_voxels_obj_with_pad
    #
    # for data in progress:
    #     pass
        # key, mesh_file_name, mesh_name, folder_name, marched_sdf, combined_uncertainty_reshaped, gt_sdf_latent_codes = data

        # out_file_name = os.path.join(res_dir + "_combined_sdf_ObjID_ " + str(key) + ".obj")
        # marched_sdf = marched_sdf.squeeze(0)
        # make_mcubes_from_voxels_obj_with_pad(marched_sdf, out_file_name)

# #
# if __name__ == "__main__":
#     TestLMDB_pytorch_dataloader()

