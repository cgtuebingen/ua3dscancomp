import os
import sys

sys.path.append("..")
from train import CompletePartialScans
from pytorch_lightning.strategies import DDPStrategy
import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor


def main(eval_root: str):
    parser = argparse.ArgumentParser()
    #  for SDFtoSDF
    parser.add_argument("--latent_dim", default=512, type=int)  # 512
    parser.add_argument("--resolution", default=128, type=int)
    parser.add_argument("--target_resolution", default=32, type=int)

    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)

    parser.add_argument("--learning_rate", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.02, type=float)
    # --------------------------------------------
    #
    parser.add_argument(
        "--train_lmdb_path",
        default="/path_to_train_lmdb/_train_combined/",  # dataset for full mesh with 128^3
        type=str,
    )

    parser.add_argument(
        "--val_lmdb_path",
        default="/path_to_validation_lmdb/_val_withLatentCodes__0_1909.mdb",  # dataset for full mesh with 128^3
        type=str,
    )

    parser.add_argument(
        "--test_lmdb_path",
        default="/path_to_test_lmdb/_test_withLatentCodes__0_5000.mdb",  # dataset for full mesh with 128^3
        type=str,
    )
    parser.add_argument(
        "--mesh_path",
        default="/path_to_datasets_objaverse1.0_processed/",
        type=str,
    )

    parser.add_argument("--value_range", default=1, type=int)
    # on ceph
    parser.add_argument(
        "--vae_checkpoint_path",
        default="/path_to_vae_checkpoint/",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--marching_cube_result_dir",
        default="/path_to_GT_data/marching_cube_result_dir/",
        type=str,
    )

    # hparams for transformer
    parser.add_argument(
        "--layers", default=18, type=int
    )  # layers: Number of transformer layers.
    parser.add_argument(
        "--dim_size", default=512 * 4, type=int
    )  # Dimensionality of latent space in transformer.
    parser.add_argument(
        "--heads", default=16, type=int
    )  # heads: Number of attention heads.
    parser.add_argument("--pre_trained", default=True, type=bool)

    parser.add_argument("--image_resolution", default=256, type=int)

    parser.add_argument("--resume_on_previous_model", type=bool, default=False)
    parser.add_argument("--previous_model_ckpt_path", type=str, default="")
    # test and eval:
    parser.add_argument("--num_samples", default=1000000, type=int)
    parser.add_argument(
        "--num_views_for_test", required=True, type=int
    )

    # num_gpus = 3
    # num_train_steps = len(train_dataset) // (batch_size * num_gpus) * trainer.max_epochs
    # print("\n num_train_steps: ", num_train_steps)
    # num_warmup_steps = int(hparams.warmup_ratio * num_train_steps)
    parser.add_argument("--num_warmup_steps", required=True, type=int)
    parser.add_argument("--num_train_steps", required=True, type=int)

    args = parser.parse_args()

    obj_dir = os.path.join(
        eval_root, "obj_dir" + "_num_views-" + str(args.num_views_for_test) + "/"
    )
    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)
    # just for rendering
    rendered_obj_dir = os.path.join(
        eval_root,
        "rendered_obj_dir" + "_num_views-" + str(args.num_views_for_test) + "/",
    )
    if not os.path.isdir(rendered_obj_dir):
        os.mkdir(rendered_obj_dir)

    eval_dir = os.path.join(
        eval_root, "eval_dir" + "_num_views-" + str(args.num_views_for_test) + "/"
    )
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)
    # write the checkpoints every 1000 steps
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        filename="checkpoint-{epoch:03d}-{loss:.3f}",
        save_top_k=4,
        save_last=True,
        mode="min",
        verbose=True,
        every_n_train_steps=1000,
    )
    lr_Monitor = LearningRateMonitor(logging_interval="step")
    model = CompletePartialScans(
        latent_dim=args.latent_dim,
        resolution=args.resolution,
        target_resolution=args.target_resolution,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        train_lmdb_path=args.train_lmdb_path,
        val_lmdb_path=args.val_lmdb_path,
        test_lmdb_path=args.test_lmdb_path,
        mesh_path=args.mesh_path,
        value_range=args.value_range,
        vae_checkpoint_path=args.vae_checkpoint_path,
        marching_cube_result_dir=args.marching_cube_result_dir,
        layers=args.layers,
        dim_size=args.dim_size,
        heads=args.heads,
        pre_trained=args.pre_trained,
        image_resolution=args.image_resolution,
        resume_on_previous_model=args.resume_on_previous_model,
        previous_model_ckpt_path=args.previous_model_ckpt_path,
        eval_dir=eval_dir,
        obj_dir=obj_dir,
        num_samples=args.num_samples,
        num_views_for_test=args.num_views_for_test,
        num_warmup_steps=args.num_warmup_steps,
        num_train_steps=args.num_train_steps,
    )

    # configure the pytorch-lightning trainer.
    trainer = pl.Trainer(
        # args,
        accelerator="gpu",
        devices=-1,
        num_nodes=1,
        strategy=DDPStrategy(),
        max_epochs=14,  # 14
        log_every_n_steps=500,
        # detect_anomaly=True,
        callbacks=[checkpoint_callback, lr_Monitor],
        val_check_interval=10000,
        default_root_dir="/path_to_tensorboard_log_root/",
        precision="bf16-mixed",
    )
    trainer.fit(model)
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.nccl.version()", torch.cuda.nccl.version())
    # torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")

    torch.set_float32_matmul_precision("high")

    eval_root = "/path_to_eval_root/"
    main(eval_root)
