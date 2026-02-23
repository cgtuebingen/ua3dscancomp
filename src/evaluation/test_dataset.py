import os
import sys

sys.path.append("..")


import numpy as np
import pytorch_lightning as pl
import msgpack
import msgpack_numpy as m
import torch

m.patch()
from utils.sdf_generatings_fns import setup_cu3d

from utils.partial_view_generation_fns import (
    compute_partial_views,
    create_cuda_raster_context,
)
import utils.shared_dataset_fns as data_fn
import utils.shared_lmdb_fns as lmdb_fn


# -------------------------------------------------------------------------------------------------------------------------------------
class TESTLMDBOBJAVERSEPARTIALVIEWS(pl.LightningDataModule):
    def __init__(
        self,
        mesh_dir: str,
        views_dict_path: str,
        lmdb_path: str,
        res_dir: str,
        image_resolution: int,
        num_views: int,
        resolution: int,
        device="cuda:0",
    ):

        super(TESTLMDBOBJAVERSEPARTIALVIEWS).__init__()

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

        views_dict = torch.load(views_dict_path)

        self.mvps = views_dict["mvps"]
        self.view_dirs = views_dict["view_dirs"]
        self.shuffled_views = views_dict["shuffled_views"]

        self.my_lmdb = None
        self.num_views = num_views

        self.len: int
        if os.path.isdir(lmdb_path):  # if the database exists already:
            my_lmdb = lmdb_fn.openLMDB(lmdb_path)
            with my_lmdb.begin(write=False) as lmdb_txn:  # read it
                self.keys = msgpack.unpackb(lmdb_txn.get(b"__keys__"))  # list of keys
                self.len = len(self.keys)
        else:  # if it does not exist
            raise RuntimeError("LMDB DOES NOT EXIST!")

    def __len__(self):
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
            self.my_lmdb = lmdb_fn.openLMDB(
                self.lmdb_path
            )  # create an object and open the database
            self.setup_cuda_stuff()

        if idx < 0 or idx is None:
            raise "invalid item index"

        key = self.keys[idx]

        with self.my_lmdb.begin(
            write=False
        ) as lmdb_txn:  # reading what is written before using the object
            raw_example = msgpack.unpackb(lmdb_txn.get(msgpack.packb(key)))
            mesh_file_name = raw_example["mesh_file_name"]
            mesh_name = raw_example["mesh_name"]
            folder_name = raw_example["folder_name"]
            faces = np.array(raw_example["faces"])
            vertices = np.array(raw_example["vertices"])

            gt_sdf_latent_codes = np.array(
                raw_example["marched_sdf_latent_codes"], copy=True
            )

        vertices = torch.from_numpy(vertices).to(device=self.device)
        faces = torch.from_numpy(faces).to(dtype=torch.int32).to(device=self.device)

        num_views = self.num_views

        data = data_fn.init_combined_data(num_views)

        # pick views from shuffled views----------------------------------------------------------------------------------------------------------------------------------------------------------------
        assert num_views <= len(self.view_dirs)
        random_views = self.shuffled_views[key, 0:num_views]
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
        combined_sdf_reshaped = combined_sdf.reshape(
            [self.resolution, self.resolution, self.resolution]
        )

        marched_dict = data_fn.do_sdf_on_marched_sdf(
            self.cu3d_instance, combined_sdf_reshaped.cpu().numpy(), self.device
        )
        if marched_dict is None:  # check if the marched ones is empty. so we skip it.
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
        ) - 1  # the input is [0, 100], -> normalize to [-1, 1]

        return [
            key,
            mesh_file_name,
            mesh_name,
            folder_name,
            marched_sdf,
            combined_uncertainty_reshaped_normalized,
            vertices,
            faces,
            gt_sdf_latent_codes,
        ]
