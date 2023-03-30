import copy

import open3d as o3d
import zipfile
import tempfile
from pathlib import Path
import numpy as np
import os
from scipy.spatial.transform import Rotation

class Showable():
    def __init__(self, file=None, mesh=None):
        self.file = file
        self.mesh = mesh
        self.transform_matrix = np.eye(4)
        if mesh is None:
            self.requires_generation = True
        else:
            self.requires_generation = False

    @property
    def transformed_mesh(self):
        return copy.deepcopy(self.mesh).transform(self.transform_matrix)

    def get_showable(self):
        if self.mesh is not None:
            return self.transformed_mesh
        else:
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh = mesh.transform(self.transform_matrix)
            return mesh

    def center_mesh(self):
        if self.mesh is not None:
            center = self.mesh.get_center()
        else:
            raise ValueError("Object must have a mesh to move mesh center to absolute location.")
        coords2 = []
        for ce in center:
            coords2.append(-ce)
        coords = coords2
        xform1 = np.eye(4)
        xform1[:3, -1] = coords
        self.transform_matrix = np.matmul(xform1, self.transform_matrix)
        return self

    def translate(self, coords):
        if len(coords) == 3:
            xform1 = np.eye(4)
            xform1[:3, -1] = coords
            self.transform_matrix = np.matmul(xform1, self.transform_matrix)
        else:
            raise ValueError("coordinate variable should be a tuple or list of length 3")
        return self

    def rotate(self, matrix3x3, center=True):
        if isinstance(matrix3x3, Rotation):
            if not matrix3x3.single:
                for r in matrix3x3:
                    self.rotate(r.as_matrix(), center=center)
            else:
                self.rotate(matrix3x3.as_matrix(), center=center)
            return self
        if center == True:
            original_center = self.transform_matrix[:3, -1]
            self.transform_matrix[:3, -1] = 0
            rot4 = np.eye(4)
            rot4[:3, :3] = matrix3x3
            self.transform_matrix = np.matmul(rot4, self.transform_matrix)
            self.transform_matrix[:3, -1] = original_center
        else:
            rot4 = np.eye(4)
            rot4[:3, :3] = matrix3x3
            self.transform_matrix = np.matmul(rot4, self.transform_matrix)
        return self

    def scale(self, s):
        s4 = np.eye(4)
        s4[4, 4] = s
        self.transform_matrix = np.matmul(s4, self.transform_matrix)
        return self

    def transform(self, matrix4x4):
        self.transform_matrix = np.matmul(matrix4x4, self.transform_matrix)
        return self


def load(filename):
    with open(filename, 'r') as file:

        pname = Path(file.name)
        ext = pname.suffix
        stem = pname.stem

        if ext == '.zmarch':
            archive = zipfile.ZipFile(file.name, 'r')
            tmp_dir = tempfile.mkdtemp()
            archive.extractall(tmp_dir)
            mesh = o3d.io.read_triangle_mesh(tmp_dir + os.sep + stem + '.ply')
            showable = Showable(file.name, mesh)
            archive.close()
        elif ext == '.march':
            showable = Showable(file.name)
        elif ext in ['.ply','.stl','.obj','.off','.gltf']:
            mesh = o3d.io.read_triangle_mesh(file.name)
            showable = Showable(file.name, mesh)

    return showable