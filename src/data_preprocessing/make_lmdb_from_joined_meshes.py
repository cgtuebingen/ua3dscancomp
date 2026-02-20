import os
import sys
import pytorch_lightning as pl
import lmdb
import msgpack
import msgpack_numpy as m

m.patch()
sys.path.append("..")

from pytorch3d.io import IO
import trimesh
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from tqdm import tqdm
import numpy as np


# -------------------------------------------------------------------------------------------------------------------------------------
class DumpObjavarseIntoLmdbChunk(pl.LightningDataModule):
    def __init__(self, mesh_list: list, lmdb_path: str, mesh_dir: str):

        super(DumpObjavarseIntoLmdbChunk).__init__()

        self.lmdb_path = lmdb_path

        self.my_lmdb = None
        self.my_lmdb = self.create_new_lmdb(lmdb_path, mesh_dir)

        self.mesh_list = mesh_list

        index = 0
        self.keys = []
        self.processed_mesh_file_names = []
        for i in tqdm(range(len(self.mesh_list)), desc="Processing Meshes"):
            mesh_file_current_path = self.mesh_list[i]
            if mesh_file_current_path.endswith(".glb"):
                mesh_current = self.load_mesh_with_py3d(mesh_file_current_path)

                if isinstance(mesh_current, list):
                    if len(mesh_current) == 0:
                        print("\ntrimesh returned empty list")
                        continue
                    elif mesh_current is None:
                        print("\n current mesh: ", mesh_current)
                        print("\n current mesh is None")
                        continue
                # extract and put them to numpy arrays on cpu for serialization in lmdb
                bbx = (mesh_current.get_bounding_boxes()[0]).cpu().numpy()
                if bbx is None:
                    print("\n bbx is None!")
                    continue
                faces = (mesh_current.faces_list()[0]).cpu().numpy()
                if faces is None:
                    print("\n no faces!")
                    continue
                vertices = (mesh_current.verts_list()[0]).cpu().numpy()
                if vertices is None:
                    print("\n no vertices!!")
                    continue

                # open LMDB and put it into lmdb
                mesh_name = mesh_file_current_path.rsplit("/")[-1]
                folder_name = mesh_file_current_path.rsplit("/")[-2]
                example = {
                    "mesh_file_name": mesh_file_current_path,
                    "folder_name": folder_name,
                    "mesh_name": mesh_name,
                    "faces": faces,
                    "vertices": vertices,
                    "bbx": bbx,
                }
                with self.my_lmdb.begin(write=True) as lmdb_txn:
                    example_b = msgpack.packb(example, default=m.encode)
                    del example
                    index_b = msgpack.packb(index)
                    lmdb_txn.put(index_b, example_b)
                    del example_b, index_b
                    self.processed_mesh_file_names.append(mesh_file_current_path)
                    self.keys.append(index)
                    index += 1
                    # commit
                    if index % 128 == 0:
                        self.my_lmdb.sync()
        #  write mesh files that are processed
        out_name = os.path.join(self.lmdb_path, "processed_glb_file_names.txt")
        with open(out_name, "w") as out:
            # Iterating over each element of the list
            for line in self.processed_mesh_file_names:
                out.write(line)  # Adding the line to the text.txt
                out.write("\n")  # Adding a new line character
        with self.my_lmdb.begin(write=True) as lmdb_txn:
            lmdb_txn.put(b"__keys__", msgpack.packb(self.keys))

    def load_mesh_with_py3d(self, glb_path: str):
        io = IO()
        io.register_meshes_format(MeshGlbFormat())
        mesh = io.load_mesh(glb_path, include_textures=False)
        return mesh

    def create_new_lmdb(self, lmdb_path: str, mesh_dir: str):
        assert not (os.path.isdir(lmdb_path))  # if the database exists already:
        # if it does not exist
        # write it:
        my_lmdb = lmdb.open(
            lmdb_path,
            readonly=False,
            lock=False,
            readahead=True,
            map_size=32 * 1024 * 1024 * 1024 * 1024,
            sync=False,
            metasync=False,
        )
        with my_lmdb.begin(write=True) as lmdb_txn:
            lmdb_txn.put(b"__mesh_dir__", msgpack.packb(mesh_dir))

        return my_lmdb

    def __len__(self):
        return len(self.keys)

    def openLMDB(self, path: str):
        my_lmdb = lmdb.open(
            path,
            max_dbs=2,
            readonly=True,  # we just want to read it
            lock=False,  # reading!!
            readahead=True,
            map_size=32 * 1024 * 1024 * 1024,
            max_readers=10000,
        )
        my_lmdb.open_db()
        return my_lmdb

    def __getitem__(self, idx: int):
        if self.my_lmdb is None:  # if database object is none
            self.my_lmdb = self.openLMDB(
                self.lmdb_path
            )  # create an object and open the database

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
            faces = np.array(raw_example["faces"], copy=True)
            vertices = np.array(raw_example["vertices"])
            bbx = np.array(raw_example["bbx"])

        return [key, mesh_file_name, mesh_name, folder_name, faces, vertices, bbx]


def ObjaverseToLMDB(stage: str):
    mesh_dir = "/graphics/scratch2/datasets/objaverse1.0_processed/"
    lmdb_path = "/graphics/scratch2/datasets/objaverse1.0_processed/filtered_objaverse_joined_lmdb/"

    # # read the list of meshes-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    with open(
        "/graphics/scratch2/datasets/objaverse1.0_processed/filtered_objaverse_joined_list.txt",
        "r",
    ) as f:
        filtered_objaverse_joined_list = f.read()
        filtered_objaverse_joined_list = filtered_objaverse_joined_list.split("\n")[:-1]
    assert len(filtered_objaverse_joined_list) == 731909
    # # define splits-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    mesh_list_train = filtered_objaverse_joined_list[
        0:725000
    ]  # 725000 many meshes for train
    mesh_list_test = filtered_objaverse_joined_list[
        725000:730000
    ]  # 5000 many meshes for test/evaluation
    mesh_list_val = filtered_objaverse_joined_list[
        730000:731909
    ]  # 1909 many meshes for validation while training
    # # train split---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    assert mesh_list_train[0] == filtered_objaverse_joined_list[0]
    assert mesh_list_train[1] == mesh_list_train[0 + 1]
    assert mesh_list_train[-1] == mesh_list_train[725000 - 1]
    print("\n len mesh_list_train: ", len(mesh_list_train))

    with open(
        "/graphics/scratch2/datasets/objaverse1.0_processed/filtered_objaverse_joined_list_train.txt",
        "w",
    ) as f:
        for line in mesh_list_train:
            f.write(f"{line}\n")
    print("\n train writing is done!")
    # test split----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    assert mesh_list_test[0] == filtered_objaverse_joined_list[725000]
    assert mesh_list_test[1] == filtered_objaverse_joined_list[725000 + 1]
    assert mesh_list_test[-1] == filtered_objaverse_joined_list[730000 - 1]
    print("\n len mesh_list_test: ", len(mesh_list_test))

    with open(
        "/graphics/scratch2/datasets/objaverse1.0_processed/filtered_objaverse_joined_list_test.txt",
        "w",
    ) as f:
        for line in mesh_list_test:
            f.write(f"{line}\n")
    print("\n test writing is done!")

    # validation split----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    assert mesh_list_val[0] == filtered_objaverse_joined_list[730000]
    assert mesh_list_val[1] == filtered_objaverse_joined_list[730000 + 1]
    assert mesh_list_val[-1] == filtered_objaverse_joined_list[731909 - 1]
    print("\n len mesh_list_val: ", len(mesh_list_val))

    with open(
        "/graphics/scratch2/datasets/objaverse1.0_processed/filtered_objaverse_joined_list_val.txt",
        "w",
    ) as f:
        for line in mesh_list_val:
            f.write(f"{line}\n")
    print("\n validation writing is done!")

    if stage == "train":

        train_lmdb_path = os.path.join(lmdb_path, "_train")
        train_dataset = DumpObjavarseIntoLmdbChunk(
            mesh_list_train, train_lmdb_path, mesh_dir
        )
        print("\n train dataset len:", len(train_dataset))

        for i in tqdm(range(len(train_dataset)), desc="Train Testing Dataset Samples"):
            sample = train_dataset[i]
            key, mesh_file_name, mesh_name, folder_name, faces, vertices, bbx = sample
        print("\n Train testing is Done!")

    elif stage == "test":

        test_lmdb_path = os.path.join(lmdb_path, "_test")
        test_dataset = DumpObjavarseIntoLmdbChunk(
            mesh_list_test, test_lmdb_path, mesh_dir
        )
        print("\n test dataset len:", len(test_dataset))

        for i in tqdm(range(len(test_dataset)), desc="Test Testing Dataset Samples"):
            sample = test_dataset[i]
            key, mesh_file_name, mesh_name, folder_name, faces, vertices, bbx = sample
        print("\n Test testing is Done!")

    elif stage == "val":
        val_lmdb_path = os.path.join(lmdb_path, "_val")
        val_dataset = DumpObjavarseIntoLmdbChunk(mesh_list_val, val_lmdb_path, mesh_dir)
        print("\n val dataset len:", len(val_dataset))

        for i in tqdm(
            range(len(val_dataset)), desc="Validation Testing Dataset Samples"
        ):
            sample = val_dataset[i]
            key, mesh_file_name, mesh_name, folder_name, faces, vertices, bbx = sample
        print("\n Val testing is Done!")

    else:
        print("\n the stage is invalid!")


if __name__ == "__main__":
    ObjaverseToLMDB(stage="train")
