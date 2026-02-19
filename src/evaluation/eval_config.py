import os
import sys
sys.path.append('./')
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
    parser.add_argument(
        "--train_lmdb_path",
        default="/path_to_train_lmdb/_train_combined/",  # dataset for full mesh with 128^3
        type=str,
        required=True
    )

    parser.add_argument(
        "--val_lmdb_path",
        default="/path_to_validation_lmdb/_val_withLatentCodes__0_1909.mdb",  # dataset for full mesh with 128^3
        type=str,
        required=True
    )

    parser.add_argument(
        "--test_lmdb_path",
        default="/path_to_test_lmdb/_test_withLatentCodes__0_5000.mdb",  # dataset for full mesh with 128^3
        type=str,
        required=True
    )

    parser.add_argument(
        "--mesh_path",
        default="/path_to_datasets_objaverse1.0_processed/",
        type=str,
    )

    parser.add_argument("--value_range", default=1, type=int)

    parser.add_argument(
        "--vae_checkpoint_path",
        default="/path_to_vae_checkpoint/",
        type=str,
        required=True
    )

    parser.add_argument(
        "--common_obj_dir",
        default="/path_to_GT_data/common_obj_dir/",
        type=str,
        required=True
    )

    parser.add_argument(
        "--views_dict_path",
        default="/path_to/data/views.pkl",
        type=str,
        required=True
    )

    parser.add_argument("--pre_trained", default=True, type=bool)
    parser.add_argument("--image_resolution", default=256, type=int)

    # test and eval:
    parser.add_argument("--num_samples", default=1000000, type=int)
    parser.add_argument("--num_views_for_test", type=int, required=True)

    parser.add_argument("--min_range", type=int, default=0)
    parser.add_argument("--max_range", type=int, default=5000)

    args = parser.parse_args()
    #
    obj_dir = os.path.join(eval_root, "obj_dir" + "_num_views-" + str(args.num_views_for_test) + "/")
    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)
    # just for rendering the results
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
            views_dict_path=args.views_dict_path,
            pre_trained=args.pre_trained,
            image_resolution=args.image_resolution,
            ckpt_path=ckpt_path,
            eval_dir=eval_dir,
            obj_dir=obj_dir,
            num_samples=args.num_samples,
            num_views_for_test=args.num_views_for_test,
            min_range=args.min_range,
            max_range=args.max_range,
        )

    model.test()
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    ckpt_path = "path_to_shape_completion_checkpoint"

    eval_root = "/path_to_eval_roo/"
    eval_dir = os.path.join(eval_root, 'test')
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    main_test(eval_dir, ckpt_path=ckpt_path)
