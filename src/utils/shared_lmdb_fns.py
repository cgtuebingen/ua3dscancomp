import os
import sys
import lmdb
import msgpack
import msgpack_numpy as m

m.patch()
sys.path.append("..")


def openLMDB(path: str, subdir: bool = True):
    my_lmdb = lmdb.open(
        path,
        # max_dbs=2,
        readonly=True,  # we just want to read it
        lock=False,  # reading!!
        readahead=True,
        map_size=32 * 1024 * 1024 * 1024,
        subdir=subdir,
        # max_readers=10000,
    )
    my_lmdb.open_db()
    return my_lmdb


def create_new_lmdb(lmdb_path: str, mesh_dir: str):
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
