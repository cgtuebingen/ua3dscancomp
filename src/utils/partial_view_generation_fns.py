import sys

sys.path.append("..")

import numpy as np
import torch
from Nvdiffrast.samples.torch import util
import Nvdiffrast.nvdiffrast.torch as dr
from typing import Union, Tuple
import torch.linalg as LA


# ----------------------------------------------------------------------------
# Transform vertex positions to clip space
def transform_pos(mtx: torch.Tensor, pos: torch.Tensor):
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1], device=mtx.device)], dim=1)
    return torch.matmul(posw, mtx.transpose(1, 2))


def rasterize(
    glctx, mtx: torch.Tensor, pos: torch.Tensor, pos_idx: torch.Tensor, resolution: int
):
    pos_clip = transform_pos(mtx, pos)
    # (u, v, z/w, triangle_id)
    rast_out, _ = dr.rasterize(
        glctx, pos_clip, pos_idx, resolution=[resolution, resolution]
    )
    return pos_clip, rast_out


def shade(pos_clip, rast_out, pos_idx, vtx_col, col_idx):
    color, _ = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color


def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n // ncols
    assert n == nrows * ncols
    return (
        arr.reshape(nrows, ncols, height, width, nc)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, nc)
    )


def triangulate_on_grid(width, height, device: str):
    # counter clockwise (assuming image top-down)
    # faces_init_1 = [[0+x, width+x, 1+x] for x in range(width -1)]
    # faces_init_2 = [[x+width, (x + 1)+width, 1+x] for x in range(width - 1)]

    # counter clockwise (assuming image bottom-up)
    faces_init_1 = [[1 + x, width + x, 0 + x] for x in range(width - 1)]
    faces_init_2 = [[1 + x, (x + 1) + width, x + width] for x in range(width - 1)]

    faces_init = list(zip(faces_init_1, faces_init_2))

    faces_init = torch.tensor(faces_init)
    faces_rows = torch.cat([faces_init + width * y for y in range(height - 1)])
    faces_rows = torch.flatten(faces_rows, start_dim=0, end_dim=1)
    return faces_rows


def exclude_invalid_triangle_id(
    triangle_id: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    tri = torch.flatten(triangle_id, start_dim=1, end_dim=2)
    valid_triangle = tri > 0
    tri_v = valid_triangle[:, faces[:, :]]
    indices_to_keep = torch.all(tri_v, dim=2)

    return indices_to_keep


def calculate_surface_normals(
    vertices: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    v0 = vertices[:, faces[:, 0]]
    v1 = vertices[:, faces[:, 1]]
    v2 = vertices[:, faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    face_normals = torch.cross(e1, e2, dim=2)
    normalized_face_normals = torch.nn.functional.normalize(face_normals, dim=2)
    return normalized_face_normals


def clean_faces_based_on_depth(cam_direction, vertices, faces_to_keep) -> torch.bool:
    normalized_face_normals = calculate_surface_normals(vertices, faces_to_keep)

    # cam_direction = R[:, :, 2].unsqueeze(1)
    # dot_product = torch.sum(normalized_face_normals * cam_direction, dim=2)
    dot_product = LA.vecdot(normalized_face_normals, cam_direction, dim=2)
    # dot_product_condition = dot_product > 0.5  # 60degrees=0.5, 75degrees=0.25, 89.5degrees=0.005
    dot_product_condition = -dot_product > 0.25
    # dot_product_condition = dot_product < -0.5

    return dot_product_condition


def create_cuda_raster_context(device: str = "cuda:0") -> dr.RasterizeCudaContext:
    # Rasterizer context
    # glctx = dr.RasterizeGLContext() if device == 'OpenGL' else dr.RasterizeCudaContext(device=device)
    return dr.RasterizeCudaContext(device=device)


def generate_mvp_matrices(num_views: int) -> Tuple[torch.Tensor, torch.Tensor]:

    # spherical fibonacci
    cosTheta = np.linspace(1, -1, num_views, endpoint=False) - 1 / (2 * num_views)
    cosTheta = torch.from_numpy(cosTheta)

    # golden_ratio = (np.sqrt(5.0) + 1) * 0.5
    golden_ratio = (np.sqrt(5.0 + 1)) * 0.5
    phi = np.linspace(
        0, 2.0 * np.pi * num_views / golden_ratio, num_views, endpoint=False
    )
    phi = torch.from_numpy(phi)

    sinTheta = torch.sqrt(1.0 - cosTheta * cosTheta)
    cosPhi = torch.cos(phi)
    sinPhi = torch.sin(phi)

    x, y, z = sinTheta * cosPhi, sinTheta * sinPhi, cosTheta

    dist = 8.0
    cameraPos = torch.stack([x, y, z], dim=1) * dist
    cameraPos = cameraPos.to(dtype=torch.float32)

    target = torch.as_tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    up = torch.as_tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    homo = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    # "x" sets the fov
    proj = torch.from_numpy(util.projection(x=0.2)).to(dtype=torch.float32)

    # rotate into camera coordinate system
    dir = torch.nn.functional.normalize(target.unsqueeze(0) - cameraPos, dim=1)
    right = torch.nn.functional.normalize(
        torch.linalg.cross(dir, up.unsqueeze(0)), dim=1
    )
    new_up = torch.linalg.cross(right, dir, dim=1)

    # already transposed
    rotation = torch.stack([right, new_up, -dir], dim=1)
    # translate camera to 0
    translation = torch.bmm(rotation, -cameraPos.unsqueeze(2))

    # model -> view
    mv = torch.cat([rotation, translation], dim=2)
    mv = torch.cat(
        [mv, homo.unsqueeze(0).unsqueeze(0).expand([num_views, 1, 4])], dim=1
    )

    # project
    mvp = torch.matmul(proj, mv)

    return mvp, dir


def compute_partial_views(
    vertices: torch.Tensor,
    face_indices: torch.Tensor,
    glctx: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext],
    mvps: torch.Tensor,
    dirs: torch.Tensor,
    resolution: int = 64,
    device: str = "cuda:0",
):

    # (u, v, z/w, triangle_id)
    # rast_out
    _, rast_out = rasterize(glctx, mvps, vertices, face_indices, resolution)

    triangle_id = rast_out[:, :, :, 3]
    world_coords, _ = dr.interpolate(vertices.unsqueeze(0), rast_out, face_indices)
    del vertices, face_indices, rast_out

    vertices = torch.flatten(world_coords, start_dim=1, end_dim=2)

    # triangulate on the grid
    faces_rows = triangulate_on_grid(resolution, resolution, device).to(device=device)
    face_mask_to_keep = exclude_invalid_triangle_id(triangle_id, faces_rows)
    dot_product_condition = clean_faces_based_on_depth(
        dirs.unsqueeze(1).to(device=device), vertices, faces_rows
    )

    both_conditions = torch.logical_and(face_mask_to_keep, dot_product_condition)

    meshes_from_vertices = []
    for i in range(mvps.shape[0]):
        faces_to_keep = faces_rows[both_conditions[i, :]]
        if len(faces_to_keep) == 0:
            # print('\n nothing visible in view ', i)
            continue

        mesh_from_vertices = (vertices[i], faces_to_keep)
        meshes_from_vertices.append(mesh_from_vertices)

    return meshes_from_vertices
