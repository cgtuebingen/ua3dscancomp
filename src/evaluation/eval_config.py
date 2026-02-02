import os
import sys
sys.path.append('/home/zakeri/Documents/Codes/MyCodes/Proposal2/ua3dscancomp-gitbub/src/')
import argparse
from eval import EvalObjaverse


def main_test(eval_root: str, ckpt_path: str):
    parser = argparse.ArgumentParser()
    #  for SDFtoSDF

    parser.add_argument("--latent_dim", default=512, type=int)  # 512
    parser.add_argument("--resolution", default=128, type=int)
    parser.add_argument("--target_resolution", default=32, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)

    # --------------------------------------------
    # on Heracleum or cluster-gpu-02
    parser.add_argument(
        "--train_lmdb_path",
        default="/graphics/scratch3/staff/zakeri/LMDBs/filtered_objaverse_joined_lmdb/_train_combined/",  # dataset for full mesh with 128^3
        type=str,
    )

    parser.add_argument(
        "--val_lmdb_path",
        default="/graphics/scratch3/staff/zakeri/LMDBs/filtered_objaverse_joined_lmdb/_val_withLatentCodes__0_1909.mdb",  # dataset for full mesh with 128^3
        type=str,
    )

    parser.add_argument(
        "--test_lmdb_path",
        default="/graphics/scratch3/staff/zakeri/LMDBs/filtered_objaverse_joined_lmdb/_test_withLatentCodes__0_5000.mdb",  # dataset for full mesh with 128^3
        type=str,
    )

    parser.add_argument(
        "--mesh_path",
        default="/graphics/scratch2/datasets/objaverse1.0_processed/",
        type=str,
    )

    parser.add_argument("--value_range", default=1, type=int)

    # parser.add_argument(
    #     "--vae_checkpoint_path",
    #     default="/graphics/scratch2/staff/zakeri/train_logs/VAE/skip_connection/v403_64_2x2x2_noBNDecoder_shapenetcorev2_excluding_shapenetcorev1_validation_split/lightning_logs/version_0/checkpoints/saved/checkpoint-epoch=193-loss=0.000.ckpt/",
    #     type=str,
    # )
    parser.add_argument(
        "--vae_checkpoint_path",
        default="/graphics/scratch3/staff/zakeri/VAE_Checkpoint/checkpoint-epoch=193-loss=0.000.ckpt/",
        type=str,
    )

    parser.add_argument(
        "--common_obj_dir",
        default="/graphics/scratch3/staff/zakeri/ObjaverseEval/common_obj_dir/",
        type=str,
    )

    parser.add_argument("--pre_trained", default=True, type=bool)
    parser.add_argument("--image_resolution", default=256, type=int)

    # test and eval:
    parser.add_argument("--num_samples", default=1000000, type=int)
    parser.add_argument("--num_views_for_test", type=int, default=1)  # TODO required

    parser.add_argument("--min_range", type=int,default=0)  # required=True
    parser.add_argument("--max_range", type=int, default=5000)
    parser.add_argument("--rand_rotation_angle", type=float, default=1) # TODO required

    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    #
    obj_dir = os.path.join(eval_root, "obj_dir" + "_num_views-" + str(args.num_views_for_test) + "/")
    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)
    # just for rendering
    rendered_obj_dir = os.path.join(eval_root, "rendered_obj_dir" + "_num_views-" + str(args.num_views_for_test) + "/")
    if not os.path.isdir(rendered_obj_dir):
        os.mkdir(rendered_obj_dir)

    eval_dir = os.path.join(eval_root, "eval_dir" + "_num_views-" + str(args.num_views_for_test) + "/")
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    model = EvalObjaverse(
            latent_dim=args.latent_dim,
            resolution=args.resolution,
            target_resolution=args.target_resolution,
            val_batch_size=args.val_batch_size,
            test_lmdb_path=args.test_lmdb_path,
            mesh_path=args.mesh_path,
            vae_checkpoint_path=args.vae_checkpoint_path,
            common_obj_dir=args.common_obj_dir,
            pre_trained=args.pre_trained,
            image_resolution=args.image_resolution,
            ckpt_path=ckpt_path,
            eval_dir=eval_dir,
            obj_dir=obj_dir,
            num_samples=args.num_samples,
            num_views_for_test=args.num_views_for_test,
            min_range=args.min_range,
            max_range=args.max_range,
            rand_rotation_angle=args.rand_rotation_angle,
        )

    model.test()
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])

if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # print("torch.cuda.device_count()", torch.cuda.device_count())
    # torch.multiprocessing.set_sharing_strategy("file_system")

    # ckpt_path = "/graphics/scratch3/staff/zakeri/saved_training_logs/PartialScanCompletion/bf16_mixed/version_7/checkpoints/used_for_paper/checkpoint-epoch=010-loss=0.000-v1.ckpt"  # ev1
    # ckpt_path = "/ceph/zakeri/ParticalScanComletion/training_logs/bf16_mixed/lightning_logs/version_7/checkpoints/last.ckpt" # not better than checkpoint-epoch=010-loss=0.000-v1.ckpt
    # this is the checkpoint resumed from version_7/checkpoints/used_for_paper/checkpoint-epoch=010-loss=0.000-v1.ckp for harsh cases and goes to the paper
    ckpt_path = "/ceph/zakeri/ParticalScanComletion/training_logs/bf16_mixed/lightning_logs/version_10/checkpoints/used_for_paper/last.ckpt"
    # eval_root = "/graphics/scratch3/staff/zakeri/ObjaverseEval/"
    # eval_dir = os.path.join(eval_root, 'ev2_lastckpt_numview1')
    # if not os.path.isdir(eval_dir):
    #     os.mkdir(eval_dir)

    eval_root = "/graphics/scratch3/staff/zakeri/ObjaverseEval/imperfect_poses_dir_clip_2/"
    eval_dir = os.path.join(eval_root, 'ev0_lastckpt_1deg')
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    main_test(eval_dir, ckpt_path=ckpt_path)
