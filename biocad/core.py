import copy
from math import cos, sin

import numpy as np
import open3d as o3d
import open3d.cpu.pybind.geometry
import torch
from torchmcubes import marching_cubes, grid_interp
import time


# todo: create a gui with a python console in it
# todo: allow getting grids for subsections of total grids where new objects will be
#  and replacing the vert data there only


def get_grid(n=256):
    x, y, z, = torch.meshgrid(
        [torch.arange(0, n, dtype=torch.float32).cuda() / n,
         torch.arange(0, n, dtype=torch.float32).cuda() / n,
         torch.arange(0, n, dtype=torch.float32).cuda() / n]
    )

    # todo: color grid should be from torch too for speed, but doing so causes a weird error even though both
    #   grids are numerically the same.
    #  This is pretty serious, since it's the difference between the program starting up in a few milliseconds and a
    #   few seconds, making it runnable from a file vs forcing the python console. However, grid creation may be slow no
    #   matter what in some cases, so perhaps a minimal cad gui with a python console is warranted.
    x2, y2, z2 = np.mgrid[:n, :n, :n]
    x2 = (x2 / n).astype('float32')
    y2 = (y2 / n).astype('float32')
    z2 = (z2 / n).astype('float32')
    rgb = np.stack((x2, y2, z2), axis=-1)
    rgb = np.transpose(rgb, axes=(3, 2, 1, 0)).copy()

    return x, y, z, rgb


def get_poly_data(rgb, u, thresh=1.0):
    rgb = torch.from_numpy(rgb).cuda()
    verts, faces = marching_cubes(u.cuda(), thresh)
    colrs = grid_interp(rgb, verts.cuda())

    verts = verts.cpu().numpy()
    faces = faces.cpu().numpy()
    colrs = colrs.cpu().numpy()

    torch.cuda.empty_cache()

    return verts, faces, colrs


def get_geometries(verts, faces, colrs):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colrs)
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    return [mesh, wire]


def get_showable(obj_list, rgb, thresh=1.0):
    u = union(obj_list)
    return get_geometries(*get_poly_data(rgb, u, thresh))


is_closed = False
geometry_update_data = None


def func(x):
    time.sleep(1.0 / 120.0)  # thread switch so I can type, 60fps is fast enough for now
    global is_closed, geometry_update_data
    if is_closed:
        x.close()
        is_closed = False
    if geometry_update_data is not None:
        x.clear_geometries()
        for geometry in geometry_update_data:
            x.add_geometry(geometry, reset_bounding_box=False)

        geometry_update_data = None


def visualize(verts, faces, colrs):
    mesh, wire = get_geometries(verts, faces, colrs)
    o3d.visualization.draw_geometries_with_animation_callback([mesh, wire], func, window_name='Marching cubes',
                                                              width=int(1920 // 2), height=1080,
                                                              left=int(1920 // 2), top=0)


def transform(x, y, z, scale=None, translate=None, euler_rotate=None):
    tx = ty = tz = sx = sy = sz = yaw = pitch = roll = None

    if translate is not None:
        tx, ty, tz = translate

        ttx = (x - tx)
        tty = (y - ty)
        ttz = (z - tz)
    else:
        ttx = x
        tty = y
        ttz = z

    if scale is not None:
        sx, sy, sz = scale
    else:
        sx = sy = sz = 1

    if euler_rotate is not None:
        yaw, pitch, roll = euler_rotate

        # I found that new_x was the inverse of x from a 3d matrix perspective.
        # In graphics, you'd do out=S*T*R*v, but for some reason x was acting like out instead of v, so I had to do:
        #   R^-1*T^-1*S^-1*out=R^-1*T^-1*S^-1*S*T*R*v=v
        # Thus, these are the inverse transformation matrices that include euler rotation, translation, and scaling.
        # Wolfram math input, excluding translation:
        #   {{1,0,0},{0,cos\(40)c\(41),sin\(40)c\(41)},{0,-sin\(40)c\(41),cos\(40)c\(41)}}{{cos\(40)b\(41),0,-sin\(40)b\(41)},{0,1,0},{sin\(40)b\(41),0,cos\(40)b\(41)}}{{cos\(40)a\(41),sin\(40)a\(41),0},{-sin\(40)a\(41),cos\(40)a\(41),0},{0,0,1}}{{Divide[1,x],0,0},{0,Divide[1,y],0},{0,0,Divide[1,z]}}

        nx = (
                     ttx * (cos(yaw) * cos(pitch)) +
                     tty * (sin(yaw) * cos(pitch)) +
                     ttz * (-sin(pitch))
             ) / sx
        ny = (
                     ttx * ((cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll))) +
                     tty * ((sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll))) +
                     ttz * (cos(pitch) * sin(roll))
             ) / sy
        nz = (
                     ttx * ((cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll))) +
                     tty * ((sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll))) +
                     ttz * (cos(pitch) * cos(roll))
             ) / sz
    else:
        nx = ttx / sx
        ny = tty / sy
        nz = ttz / sz

    return nx, ny, nz


def show(rgb, u, thresh=1.0):
    visualize(*get_poly_data(rgb, u, thresh))


def union(obj_list):
    assert len(obj_list) > 2

    if isinstance(obj_list[0], (torch.Tensor, np.ndarray)):
        u = torch.zeros_like(obj_list[0])
    else:
        raise TypeError(f"could not get grid size with first union object of type: {type(obj_list[0])}")

    u = u.to(obj_list[0].device)

    for obj in obj_list:
        u += (
            torch.where(
                obj != 0,
                torch.divide(torch.ones_like(obj), obj),
                torch.ones_like(obj) * torch.finfo(torch.float32).max
            )
        )
    return u


def intersection(obj_list):
    assert len(obj_list) > 2

    if isinstance(obj_list[0], (torch.Tensor, np.ndarray)):
        u = torch.zeros_like(obj_list[0])
    else:
        raise TypeError(f"could not get grid size with first union object of type: {type(obj_list[0])}")

    u = u.to(obj_list[0].device)

    for obj in obj_list:
        u *= (
            torch.where(
                obj != 0,
                torch.divide(torch.ones_like(obj), obj),
                torch.ones_like(obj) * torch.finfo(torch.float32).max
            )
        )
    return u


def difference(obj_a, obj_b):
    u = (
            obj_a
            -
            torch.where(
                obj_b != 0,
                torch.divide(torch.ones_like(obj_b), obj_b),
                torch.ones_like(obj_b) * torch.finfo(torch.float32).max
            )
    )
    return u


def show_all(obj_list, rgb):
    u = union(obj_list)
    show(rgb, u)


def machine_epsilon(func=float):
    # https://en.wikipedia.org/wiki/Machine_epsilon#How_to_determine_machine_epsilon
    eps = func(1)
    while func(1) + eps != func(1):
        machine_epsilon_last = eps
        eps = func(eps) / func(2)
    return machine_epsilon_last


def central_difference_epsilon(func=float):
    # https://en.wikipedia.org/wiki/Numerical_differentiation#Step_size
    return machine_epsilon(func) ** (func(1) / func(3))


def estimate_tangent_at_t(t, fx, fy, fz, func=float):
    cde = central_difference_epsilon(func)
    dx = (fx(t + cde) - fx(t - cde)) / (2 * cde)
    dy = (fy(t + cde) - fy(t - cde)) / (2 * cde)
    dz = (fz(t + cde) - fz(t - cde)) / (2 * cde)
    tang_0 = np.asarray([dx, dy, dz])
    n = np.linalg.norm(tang_0)
    tang = tang_0 / n
    return tang


def estimate_normal_at_t(t, fx, fy, fz, func=float):
    cde = central_difference_epsilon(func)

    norm_0 = (estimate_tangent_at_t(t + cde, fx, fy, fz, func) - estimate_tangent_at_t(t - cde, fx, fy, fz, func)) / (
            2 * cde)
    me = machine_epsilon(func)
    n = np.linalg.norm(norm_0)
    if np.isclose(n, np.zeros_like(n), me*10):
        raise NotImplementedError("Inputted function is not a curve. "
                                  "Estimating normals currently only supports curves.")

    normal = norm_0 / n
    return normal


def estimate_orientation_at_t(t, fx, fy, fz, func=float, order=(1,2,3)):
    tang = estimate_tangent_at_t(t, fx, fy, fz, func)
    norm = estimate_normal_at_t(t, fx, fy, fz, func)
    cross = np.cross(tang, norm)

    rotation_matrix = np.eye(3)

    def sign(x):
        s = np.sign(x)
        if s==0:
            return 1
        else: return s

    rotation_matrix[int(abs(order[0])-1), :3] = tang*sign(order[0])
    rotation_matrix[int(abs(order[1])-1), :3] = norm*sign(order[1])
    rotation_matrix[int(abs(order[2])-1), :3] = cross*sign(order[2])

    return rotation_matrix


def estimate_transform_at_t(t, fx, fy, fz, func=float, order=(1,2,3)):
    rotation_matrix = estimate_orientation_at_t(t, fx, fy, fz, func, order)
    location = np.asarray([fx(t), fy(t), fz(t)])
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = location

    #transform_matrix[int(abs(order[0])-1), 3] = location[0]*np.sign(order[0])
    #transform_matrix[int(abs(order[1])-1), 3] = location[0]*np.sign(order[1])
    #transform_matrix[int(abs(order[2])-1), 3] = location[0]*np.sign(order[2])


    return transform_matrix


import itertools as IT
import collections.abc
from typing import List, Any


def flatten(iterable, ltypes=collections.abc.Iterable) -> List[Any]:
    # https://stackoverflow.com/a/2158562/782170
    remainder = iter(iterable)
    while True:
        try:
            first = next(remainder)
        except StopIteration:
            break
        if isinstance(first, ltypes) and not isinstance(first, (str, bytes)):
            remainder = IT.chain(first, remainder)
        else:
            yield first


def copy_along_parametric_curve(obj, fx, fy, fz, t_vals, func=float, r_order=(0,1,2)):
    objs = []
    if isinstance(obj, collections.abc.Iterable):
        obj_group = list(flatten(obj))
    else:
        obj_group = [obj]

    for t in t_vals:
        for o in obj_group:
            objs.append(copy.deepcopy(o).transform(estimate_transform_at_t(t, fx, fy, fz, func, order=r_order)))
    return objs


def place_list_along_parametric_curve(objs, fx, fy, fz, t_vals=None, func=float, r_order=(0,1,2)):
    if t_vals is not None:
        assert len(t_vals) == len(objs)
    else:
        t_vals = range(len(objs))

    objs2 = []
    for o in objs:
        if isinstance(o, collections.abc.Iterable):
            objs2.append(list(flatten(o)))
        else:
            objs2.append([o])

    for t, o1 in enumerate(objs2):
        for oi in o1:
            objs.append(oi.transform(estimate_transform_at_t(t_vals[t], fx, fy, fz, func, order=r_order)))
    return objs



if __name__ == '__main__':
    import numpy as np

    print(f'central_difference_epsilon: {central_difference_epsilon(np.float64)}')
    print(f'central_difference_epsilon: {central_difference_epsilon(np.float32)}')
    print(f'central_difference_epsilon: {central_difference_epsilon(np.float16)}')
    print(f'central_difference_epsilon: {central_difference_epsilon(float)}')


    def line(t):
        return t


    print(f'curve orientation: \n{estimate_orientation_at_t(1.0, sin, cos, sin)}')

    print(list(flatten([[1, 2, [3, 4]]])))
