import os
import numpy as np
import torch

from evaluation.eval_fns import (
    extract_files_with_given_extension_general,
    extract_broken_object_indices,
    extract_metric_results,
    pickle_all_dict_files,
)

def calculate_mean_all(eval_dir: str):
    eval_file_list = extract_files_with_given_extension_general(eval_dir, '.pkl')
    print("\n eval_file_list len: ", len(eval_file_list))

    broken_object_index_all = extract_broken_object_indices(eval_file_list, eval_dir)
    # read the data o test:
    broken_object_index_all_read = []
    broken_out_name = os.path.join(eval_dir, "broken_object_index_all.txt")
    with open(broken_out_name, "r") as file:
        for line in file:
            broken_object_index_all_read.append(int(line.rstrip("\n")))
    assert (broken_object_index_all == broken_object_index_all_read)

    pickle_all_dict_files(eval_file_list, eval_dir)
    outname = os.path.join(eval_dir, "combined_pickles")
    combined_pickles = torch.load(outname)
    broken_object_index_all_unique = np.unique(broken_object_index_all)
    extract_metric_results(eval_dir, "mean_eval_results", broken_object_index_all_unique, combined_pickles)

if __name__ == "__main__":

    eval_dir = "/path_to_eval_dir/"
    calculate_mean_all(eval_dir)