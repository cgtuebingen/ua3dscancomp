import os
import torch
from tqdm import tqdm
from pycu3d.test.evaluate_all import EVALALLMETRICS

def evaluate_all(dict_args_eval: dict, dict_args_variables: dict) -> dict:
    object_index = dict_args_eval["Object_index"]
    num_samples = dict_args_eval['num_samples']

    # voxels depaccked
    completed_voxel = dict_args_eval["Completed_voxel"]
    gt_voxel = dict_args_eval["GT_voxel"]
    # input_voxel = collected_data_dict_for_plotting["Input_voxel"]

    # objs depacked
    # partial_obj = dict_args_eval["Input_file"]
    completed_obj = dict_args_eval["Transformer_file"]

    gt_obj = dict_args_eval["GT_file"]

    # extract scales:
    hausdorff_scale = dict_args_variables["hausdorff_scale"]
    chamfer_scale = dict_args_variables["chamfer_scale"]

    # on voxels
    eval_obj = EVALALLMETRICS()
    iou, fscore = eval_obj.eval_iou_and_fscore_voxels_cud3d(completed_voxel, gt_voxel)
    if torch.isnan(torch.tensor(iou)) or torch.isnan(torch.tensor(fscore)):
        iou = -1
        fscore = -1

    # on meshes
    completed_pc = eval_obj.get_obj_return_pc(completed_obj, num_samples)
    gt_pc = eval_obj.get_obj_return_pc(gt_obj, num_samples)

    # partial pc is only used if one evaluates for uhd metric
    # partial_pc = eval_obj.get_obj_return_pc(partial_obj, num_samples)
    partial_pc = None

    if (completed_pc == None) or (gt_pc == None):  # or (partial_pc == None):

        chamfer = -1
        fscore_one_percent = -1
        uhd = -1
        hausdorff = -1
        normal_consistency = -1
        inaccurate_normals = -1
        completeness = -1
        chamfer_l1 = -1

    else:

        chamfer = eval_obj.eval_chamfer(completed_pc, gt_pc)
        chamfer = chamfer * chamfer_scale

        chamfer_l1 = eval_obj.eval_chamfer_3ds2vs(completed_pc, gt_pc)
        chamfer_l1 = chamfer_l1 * hausdorff_scale

        fscore_one_percent = eval_obj.eval_fscore_pc_cud3d(completed_pc, gt_pc, thres=0.02)  # my one percent=0.02(my side length is 2), shapenet diagonal len is 1

        if torch.isnan(torch.tensor(fscore_one_percent)) or torch.isnan(torch.tensor(fscore_one_percent)):
            fscore_one_percent = -1

        # uhd = eval_obj.eval_uhd(partial_pc, completed_pc)
        # uhd = uhd * hausdorff_scale
        # if torch.isnan(torch.tensor(uhd)) or torch.isnan(torch.tensor(uhd)):
        #     uhd = -1

        hausdorff = eval_obj.eval_hausdorff(completed_pc, gt_pc)
        hausdorff = hausdorff * hausdorff_scale

        if torch.isnan(torch.tensor(hausdorff)) or torch.isnan(torch.tensor(hausdorff)):
            hausdorff = -1

        normal_consistency, inaccurate_normals = eval_obj.eval_nc_and_in(completed_pc, gt_pc)

        if torch.isnan(torch.tensor(normal_consistency)) or torch.isnan(torch.tensor(normal_consistency)):
            normal_consistency = -1

        if torch.isnan(torch.tensor(inaccurate_normals)) or torch.isnan(torch.tensor(inaccurate_normals)):
            inaccurate_normals = -1

        completeness = eval_obj.eval_completeness(completed_pc, gt_pc, thres=0.03)

        if torch.isnan(torch.tensor(completeness)) or torch.isnan(torch.tensor(completeness)):
            completeness = -1

    # pack all the eval results together
    eval_results = {
        "iou": iou,
        "fscore": fscore,
        "chamfer": chamfer,
        "fscore1%": fscore_one_percent,
        "uhd": -1,
        "hausdorff": hausdorff,
        "normal_consistency": normal_consistency,
        "inaccurate_normals": inaccurate_normals,
        "completeness": completeness,
        "chamfer_l1": chamfer_l1,
        "object_index": object_index.detach().item(),

    }
    return eval_results

def write_evaluation_result(eval_results: dict, eval_dir: str):

    object_index = eval_results["object_index"]
    pickle_name = os.path.join(eval_dir, "eval_results_for_" + "objID=" + str(object_index) + ".pkl")
    if os.path.isdir(eval_dir):
        torch.save(
            eval_results,
            pickle_name,
        )
    else:
        raise print(eval_dir + "does not exist!")

    # # test
    eval_results_read = torch.load(pickle_name)
    assert eval_results_read["object_index"] == eval_results["object_index"]


def write_dict_eval_into_text(dict_data: dict, out_path: str, name: str) -> None:
    eval_file_name = name + ".txt"

    keys = [key for key in dict_data.keys()]

    if os.path.isdir(out_path):
        file_obj = open(os.path.join(out_path, eval_file_name), "w")
        for i in range(len(keys)):
            current_key = keys[i]
            current_value = dict_data.get(current_key)

            tmp = current_key + " = "
            tmp_v = "{:.7f}".format(current_value)
            file_obj.write(tmp + str(tmp_v))
            file_obj.write("\n--------------------------------------------")
            file_obj.write("\n")

    else:
        raise ("\n dir for writing eval data does not exist!")

def pickle_all_dict_files(eval_file_list: list, eval_dir: str) -> dict:
    eval_file_list_dict = [torch.load(i) for i in eval_file_list]
    out_name = os.path.join(eval_dir, "combined_pickles")
    torch.save(eval_file_list_dict, out_name)

def extract_files_with_given_extension_general(eval_dir: str, ext: str) -> list:
    all_files = os.listdir(eval_dir)
    print(all_files)
    file_path_list = []
    for i in range(len(all_files)):
        file = all_files[i]
        if file.endswith(ext):
            file_path_list.append(os.path.join(eval_dir, file))
    return file_path_list


def extract_broken_object_indices(eval_file_list: list, eval_dir: str):

    broken_object_index_all = []
    for i in tqdm(range(len(eval_file_list)), desc="object Indices"):
        eval_file_current = eval_file_list[i]
        eval_dict_current = torch.load(eval_file_current)
        object_index_current = eval_dict_current["object_index"]
        # now check for metrics
        iou_current = eval_dict_current['iou']
        if ((iou_current == -1) or (torch.isnan(torch.tensor(iou_current)))):
            print("iou nan")
            broken_object_index_all.append(object_index_current)

        fscore_current = eval_dict_current['fscore']
        if ((fscore_current == -1) or (torch.isnan(torch.tensor(fscore_current)))):
            print("fscore nan ")
            broken_object_index_all.append(object_index_current)

        # uhd_current = eval_dict_current['uhd']
        # if (uhd_current == -1 or (torch.isnan(torch.tensor(uhd_current)))):
        #     print("uhd -1")
        #     broken_object_index_all.append(object_index_current)
        #
        fscore_one_percent = eval_dict_current['fscore1%']
        if ((fscore_one_percent == -1) or (torch.isnan(torch.tensor(fscore_one_percent)))):
            print("fscore_one_percent nan")
            broken_object_index_all.append(object_index_current)

        hausdorff_current = eval_dict_current['hausdorff']
        if (hausdorff_current == -1 or (torch.isnan(torch.tensor(hausdorff_current)))):
            print("hausdorff_current -1")
            broken_object_index_all.append(object_index_current)

        normal_consistency_current = eval_dict_current['normal_consistency']
        if (normal_consistency_current == -1 or (torch.isnan(torch.tensor(normal_consistency_current)))):
            print("normal_consistency_current -1")
            broken_object_index_all.append(object_index_current)

        inaccurate_normals_current = eval_dict_current['inaccurate_normals']
        if (inaccurate_normals_current == -1 or (torch.isnan(torch.tensor(inaccurate_normals_current)))):
            print("inaccurate_normals_current -1")
            broken_object_index_all.append(object_index_current)

        completeness_current = eval_dict_current['completeness']
        if (completeness_current == -1 or (torch.isnan(torch.tensor(completeness_current)))):
            print("completeness_current -1")
            broken_object_index_all.append(object_index_current)

        chamfer_current = eval_dict_current['chamfer']
        if (chamfer_current == -1 or (torch.isnan(torch.tensor(chamfer_current)))):
            print("chamfer_current -1")
            broken_object_index_all.append(object_index_current)

        chamfer_l1_current = eval_dict_current['chamfer_l1']
        if (chamfer_current == -1 or (torch.isnan(torch.tensor(chamfer_l1_current)))):
            print("chamfer_l1_current -1")
            broken_object_index_all.append(object_index_current)

    print("\n num broken objects: ", len(broken_object_index_all))
    out_name = os.path.join(eval_dir, "broken_object_index_all.txt")
    with open(out_name, "w") as out:
        # Iterating over each element of the list
        for line in broken_object_index_all:
            out.write(str(line))  # Adding the line to the text.txt
            out.write('\n')  # Adding a new line character

    return broken_object_index_all

def extract_metric_results(eval_dir, output_name: str, broken_object_index_all: list, combined_pickles: dict):
    iou_all = []
    fscore_all = []
    uhd_all = []
    hausdorff_all = []
    normal_consistency_all = []
    inaccurate_normals_all = []
    completeness_all = []
    fscore_one_percent_all = []
    chamfer_all = []
    chamfer_l1_all = []

    for i in tqdm(range(len(combined_pickles)), desc="EVAL RESULTS"):
        eval_dict_current = combined_pickles[i]
        object_index_current = eval_dict_current["object_index"]
        if object_index_current in broken_object_index_all:
            continue
        else:
            iou_current = eval_dict_current['iou']
            if ((iou_current == -1) or (torch.isnan(torch.tensor(iou_current)))):
                raise ("iou nan")
            else:
                iou_all.append(iou_current)

            fscore_current = eval_dict_current['fscore']
            if ((fscore_current == -1) or (torch.isnan(torch.tensor(fscore_current)))):
                raise ("fscore nan ")
            else:
                fscore_all.append(fscore_current)

            # uhd_current = eval_dict_current['uhd']
            # if (uhd_current == -1  or (torch.isnan(torch.tensor(uhd_current)))):
            #     raise ("uhd -1")
            # else:
            #     uhd_all.append(uhd_current)

            fscore_one_percent = eval_dict_current['fscore1%']
            if ((fscore_one_percent == -1) or (torch.isnan(torch.tensor(fscore_one_percent)))):
                raise ("fscore_one_percent nan")
            else:
                fscore_one_percent_all.append(fscore_one_percent)

            hausdorff_current = eval_dict_current['hausdorff']
            if (hausdorff_current == -1  or (torch.isnan(torch.tensor(hausdorff_current)))):
                raise ("hausdorff_current -1")
            else:
                hausdorff_all.append(hausdorff_current)

            normal_consistency_current = eval_dict_current['normal_consistency']
            if (normal_consistency_current == -1 or (torch.isnan(torch.tensor(normal_consistency_current)))):
                raise ("normal_consistency_current -1")
            else:
                normal_consistency_all.append(normal_consistency_current)

            inaccurate_normals_current = eval_dict_current['inaccurate_normals']
            if (inaccurate_normals_current == -1 or (torch.isnan(torch.tensor(inaccurate_normals_current)))):
                raise ("inaccurate_normals_current -1")
            else:
                inaccurate_normals_all.append(inaccurate_normals_current)

            completeness_current = eval_dict_current['completeness']
            if (completeness_current == -1 or (torch.isnan(torch.tensor(completeness_current)))):
                raise ("completeness_current -1")
            else:
                completeness_all.append(completeness_current)

            chamfer_current = eval_dict_current['chamfer']
            if (chamfer_current == -1 or (torch.isnan(torch.tensor(chamfer_current)))):
                raise ("chamfer_current -1")
            else:
                chamfer_all.append(chamfer_current)

            chamfer_l1_current = eval_dict_current['chamfer_l1']
            if (chamfer_l1_current == -1 or (torch.isnan(torch.tensor(chamfer_l1_current)))):
                raise ("chamfer_l1_current -1")
            else:
                chamfer_l1_all.append(chamfer_l1_current)

    # convert to tensor
    iou_t = torch.tensor(iou_all)
    fscore_t = torch.tensor(fscore_all)
    # uhd_t = torch.tensor(uhd_all)
    hausdorff_t = torch.tensor(hausdorff_all)
    normal_consistency_t = torch.tensor(normal_consistency_all)
    inaccurate_normals_t = torch.tensor(inaccurate_normals_all)
    completeness_t = torch.tensor(completeness_all)
    fscore_one_percent_t = torch.tensor(fscore_one_percent_all)
    chamfer_t = torch.tensor(chamfer_all)
    chamfer_l1_t = torch.tensor(chamfer_l1_all)
    # mean
    iou_mean = torch.mean(iou_t)
    fscore_mean = torch.mean(fscore_t)
    # uhd_mean = torch.mean(uhd_t)
    hausdorff_mean = torch.mean(hausdorff_t)
    normal_consistency_mean = torch.mean(normal_consistency_t)
    inaccurate_normals_mean = torch.mean(inaccurate_normals_t)
    completeness_mean = torch.mean(completeness_t)
    fscore_one_percent_mean = torch.mean(fscore_one_percent_t)
    chamfer_mean = torch.mean(chamfer_t)
    chamfer_l1_mean = torch.mean(chamfer_l1_t)
    # write them into one file
    result_dict = {"iou_mean": iou_mean,
                   "fscore_mean": fscore_mean,
                   "uhd_mean": -1,
                   "hausdorff_mean": hausdorff_mean,
                   "chamfer_l1_mean": chamfer_l1_mean,
                   "normal_consistency_mean": normal_consistency_mean,
                   "inaccurate_normals_mean": inaccurate_normals_mean,
                   "completeness_mean": completeness_mean,
                   "fscore_one_percent_mean": fscore_one_percent_mean,
                   "chamfer_mean": chamfer_mean}
    write_dict_eval_into_text(result_dict, eval_dir, output_name)
    print("\n Done writing eval results")
