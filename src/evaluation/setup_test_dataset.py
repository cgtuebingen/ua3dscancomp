import sys
sys.path.append('/home/zakeri/Documents/Codes/MyCodes/Proposal2/ua3dscancomp-gitbub/src/')

from test_dataset import TESTLMDBOBJAVERSEPARTIALVIEWS

def setup_dataset(mesh_path, test_lmdb_path, marching_cube_result_dir, image_resolution, num_views_for_test, resolution, average_rotation_deg, device):

    test_dataset = TESTLMDBOBJAVERSEPARTIALVIEWS(mesh_path, test_lmdb_path, marching_cube_result_dir,
                                                 image_resolution, num_views_for_test, resolution, average_rotation_deg, device=device)

    print("\n test_dataset len:", len(test_dataset))
    return test_dataset
