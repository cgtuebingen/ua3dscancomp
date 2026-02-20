import os
import sys

import numpy as np
import pytorch_lightning as pl
import lmdb
import msgpack
import msgpack_numpy as m
import torch

m.patch()
sys.path.append("..")
from utils.sdf_generatings_fns import setup_cu3d

from utils.partial_view_generation_fns import (
    compute_partial_views,
    generate_mvp_matrices,
    create_cuda_raster_context,
)
import utils.shared_dataset_fns as data_fn


# -------------------------------------------------------------------------------------------------------------------------------------
class LMDBOBJAVERSEPARTIALVIEWS(pl.LightningDataModule):
    def __init__(
        self,
        mesh_dir: str,
        lmdb_path: str,
        res_dir: str,
        image_resolution: int,
        resolution: int,
        device="cuda:0",
    ):

        super(LMDBOBJAVERSEPARTIALVIEWS).__init__()

        try:
            print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
        except KeyError:
            print("CUDA_VISIBLE_DEVICES", "not set")

        self.mesh_dir = mesh_dir
        self.lmdb_path = lmdb_path
        self.res_dir = res_dir
        self.resolution = resolution

        self.image_resolution = image_resolution
        self.device = device

        self.num_views_to_generate = 20
        self.mvps, self.view_dirs = generate_mvp_matrices(self.num_views_to_generate)

        self.my_lmdb = None

        self.len: int
        if os.path.isdir(lmdb_path):  # if the database exists already:
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
                    self.keys = msgpack.unpackb(
                        lmdb_txn.get(b"__keys__")
                    )  # list of keys
                    self.len = len(self.keys)
        else:  # if it does not exist
            raise RuntimeError("LMDB DOES NOT EXIST!")

    def __len__(self):
        # print("\n dataset class len: num_views: ", self.num_views)
        return self.len

    def setup_cuda_stuff(self):
        # setup cuda stuff
        self.cu3d_instance = setup_cu3d(self.device)
        self.mvps = self.mvps.to(device=self.device)
        self.view_dirs = self.view_dirs.to(device=self.device)
        self.glctx = create_cuda_raster_context(self.device)

    def __getitem__(self, idx: int):
        if self.my_lmdb is None:  # if database object is none
            print("\n Dataloader", os.getpid(), "using device", self.device)

            self.my_lmdb = data_fn.openLMDB(
                self.lmdb_path
            )  # create an object and open the database
            self.setup_cuda_stuff()

        if idx < 0 or idx is None:
            raise "invalid item index"

        if idx > len(self.keys):
            idx = idx % len(self.keys)  # reduce the idx to the len(keys)
        key = self.keys[idx]

        # example = {"mesh_file_name": mesh_file_current, "faces": faces, "vertices": vertices, "bbx": bbx}

        with self.my_lmdb.begin(
            write=False
        ) as lmdb_txn:  # reading what is written before using the object
            raw_example = msgpack.unpackb(lmdb_txn.get(msgpack.packb(key)))

            mesh_file_name = raw_example["mesh_file_name"]
            mesh_name = raw_example["mesh_name"]
            folder_name = raw_example["folder_name"]
            faces = np.array(raw_example["faces"])
            vertices = np.array(raw_example["vertices"])
            # bbx = np.array(raw_example["bbx"])

            gt_sdf_latent_codes = np.array(
                raw_example["marched_sdf_latent_codes"], copy=True
            )
            # gt_uncertainty_voxel = np.array(raw_example["gt_uncertainty_voxel"], copy=True)

        vertices = torch.from_numpy(vertices).to(device=self.device)
        faces = torch.from_numpy(faces).to(dtype=torch.int32).to(device=self.device)

        chosen_num_views = torch.randperm(4)[:1] + 1  # randperm  produces from 0 to n-1
        num_views = chosen_num_views.item()
        data = data_fn.init_combined_data(num_views)

        # RANDOM SUBSET OF VIEW_DIRS AND MVPS based on self.num_views
        assert num_views <= len(self.view_dirs)

        mode = torch.randint(0, 3, (1,)).item()
        if (
            mode == 2 and num_views > 10
        ):  # it does not make sense to do stratification if we have more than half of total num views
            mode = 0

        if mode == 0:
            random_views = torch.randperm(len(self.view_dirs))[0:num_views]
            view_dirs = self.view_dirs[random_views]
            mvps = self.mvps[random_views]

        elif mode == 1:

            max_ = (
                len(self.view_dirs) - num_views + 1
            )  # upperbound is excluded so we need to add
            random_views = torch.randint(0, max_, (1,))
            random_views = torch.arange(num_views) + random_views.item()

            view_dirs = self.view_dirs[random_views]
            mvps = self.mvps[random_views]

        elif mode == 2:
            chunk_size = (len(self.view_dirs) + num_views - 1) // num_views
            random_views = torch.randint(0, chunk_size, (num_views,)) + torch.arange(
                0, chunk_size * num_views, chunk_size
            )
            random_views = random_views % len(self.view_dirs)

            view_dirs = self.view_dirs[random_views]
            mvps = self.mvps[random_views]

        # number of views to rasterize at once
        num_good_views = 0
        raster_batch_size = 4

        for i in range(0, len(view_dirs), raster_batch_size):
            partial_meshes = compute_partial_views(
                vertices,
                faces,
                self.glctx,
                mvps[i : i + raster_batch_size],
                view_dirs[i : i + raster_batch_size],
                self.image_resolution,
                self.device,
            )

            for partial_mesh in partial_meshes:
                data = data_fn.add_to_combined_data(
                    self.cu3d_instance, data, partial_mesh, self.resolution, self.device
                )

                num_good_views += 1

            del partial_meshes

        del vertices
        del faces

        if num_good_views <= 0:
            print("\n Warning: Skipping sampleIDX: ", idx)
            return self[torch.randint(0, len(self), (1,)).item()]

        combined_sdf, combined_uncertainty = data_fn.normalize_combined_data(
            data, self.device
        )

        if torch.any(torch.isnan(combined_sdf)) or torch.any(torch.isinf(combined_sdf)):
            breakpoint()

        if torch.any(torch.isnan(combined_uncertainty)) or torch.any(
            torch.isinf(combined_uncertainty)
        ):
            breakpoint()

        # Here, do marching on the combined sdf and then calculate the sdf again from that and then return it into training pipeline for encoding
        if True:
            combined_sdf_reshaped = combined_sdf.reshape(
                [self.resolution, self.resolution, self.resolution]
            )

            marched_dict = data_fn.do_sdf_on_marched_sdf(
                self.cu3d_instance, combined_sdf_reshaped.cpu().numpy(), self.device
            )
            if (
                marched_dict is None
            ):  # check if the marched ones is empty. so we skip it.
                print("\n Warning: Skipping sampleIDX: ", idx)
                return self[torch.randint(0, len(self), (1,)).item()]
            else:
                marched_sdf = marched_dict["marched_sdf"]

        marched_sdf = marched_sdf.cpu()
        combined_uncertainty = combined_uncertainty.cpu()

        combined_uncertainty_reshaped = combined_uncertainty.reshape(
            [self.resolution, self.resolution, self.resolution]
        )
        combined_uncertainty_reshaped_normalized = (
            combined_uncertainty_reshaped / 50
        ) - 1  # he input is [0, 100], -> normalize to [-1, 1]

        # marched_sdf_reshaped = marched_sdf.reshape([self.resolution, self.resolution, self.resolution])

        return [
            key,
            mesh_file_name,
            mesh_name,
            folder_name,
            marched_sdf,
            combined_uncertainty_reshaped_normalized,
            gt_sdf_latent_codes,
        ]
