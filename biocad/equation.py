from displayarray.frame.frame_updater import FrameUpdater
from displayarray.input import key_loop
from displayarray import display
import torch
from torchrender.pixel_shader import pixel_shader
import numpy as np
import cv2


# todo: to get lines, finding the exact 0-intercept is mandatory because 2 pixels could be -400z and 400z
#  with 0 in between. So to get the 0-intercept, you take the current pixel, the one to the right of it,
#  the one below it, and the one to the bottom right, and construct a plane going through all of those points.
#  You can then set z=0 in the plane equation to find the x&y line at zero. Then,
#  using the inverse slope and the top left pixel (0,0) you can find the line going from the current top left pixel to
#  the 0-intercept line. Then do a linear algebra system of equations to solve for the x and y point
#  where both lines meet. If 0>=x>1 and 0>=y>1, that is a valid point, and separate x tensors and y tensors should
#  be given the x and y coordinates, -1 otherwise.

def get_lines(is_equation):
    pass


# todo: run cv2.find_contours on the previous outcome (x or y) to get all the separate lines,
#  since there can be zero, one, or many per equation.

# todo: iterate over the the points in the contours and add the values from the x&y tensors to get
#  the exact point locations

# todo: iterate over the points in the contours, if x2 in contiguous points x1, x2, x3, has a distance
#  from the line from x1 to x3 that's smaller than the minimum specified distance specified by 'resolution',
#  remove x2. Repeat this process until no points are removes.

# todo: add the revolve function, 0-360, and generate a quad mesh based on the lines

# todo: convert the quad mesh into a tri mesh (trimesh.geometry.triangulate_quads)

# todo: use the python trimesh library for its boolean functions to combine meshes

# todo: use trimesh's watertight function to check if a mesh is valid for printing, highlight it (red) if not

# todo: use trimesh's stl exporter

# todo: use trimesh's vertex_defects function to find concave vs convex vertex points
# todo: combine neighboring vertices to find concave/convex regions and enumerate them

# todo: use trimesh's face ordering to be able to reference faces
# todo: show face numbers in UI
# todo: make/use face ordering algorithm on convex/concave regions

width = 1280
height = 720
img = np.zeros((height, width, 3), dtype=np.float32)

from math import pi
from torch import sin, cos, tan, abs, where


def muscle_1(y, x):
    is_equation = (
            (8 * sin(16 * x + pi / 2)) / ((x ** 2) / 4 + 1) + 2 * y ** 2 + (x ** 2) / 4
            -
            3 ** 2
    )

    is_equation_2 = (
            (8 * sin(16 * x + pi / 2)) / ((x ** 2) / 5 + 1) + 1.5 * y ** 2 + (x ** 2) / 4
            -
            3.2 ** 2
    )

    closest = where(
        abs(is_equation) < abs(is_equation_2),
        is_equation,
        is_equation_2
    )

    return closest


def muscle_2(y, x):
    is_equation = (
            (2 * sin(16 * x + pi / 2)) / ((abs(x) * 2 - 5) ** 2 + 3) + 2 * y ** 2 + (x ** 2) / 4
            -
            3 ** 2
    )

    is_equation_2 = (
            (2 * sin(16 * x + pi / 2)) / ((abs(x) * 1.5 - 4) ** 2 + 3) + 1.5 * y ** 2 + (x ** 2) / 4
            -
            3.2 ** 2
    )

    closest = where(
        abs(is_equation) < abs(is_equation_2),
        is_equation,
        is_equation_2
    )

    return closest


# instead of y=sin(x), do y-sin(x)
# for parametric, solve for t in all equations
py_equation: str = "y - torch.sin(x)"


def eq(uv, ):
    y, x = uv

    # is_equation = x**2 + y**2 - 2**2

    # is_equation = torch.tan(x ** 2) + torch.tan(y ** 2) - 0

    #is_equation = y - torch.sin(x)

    is_equation = muscle_2(y,x)

    # is_equation = eval(py_equation)

    # is_equation = x ** 2 - 2 * x * y + y ** 2 - 2 * x - 2 * y
    # is_equation = y - x
    a=50
    R= .05
    start_angle = np.pi/4
    arc = torch.arctan2(y,x-2.5)
    is_equation = (((x-2.5)**2)+(y**2))-R**2*(1+(a*arc)**2)
    #is_equation = torch.tan((a / torch.sqrt(x**2+y**2))+start_angle)*-R*x-R*y
    #is_equation = np.tan(a) - a/(np.pi/2 - torch.arctan(x/y)) - y/x

    #is_equation = torch.tan((a / torch.sqrt(x ** 2 + y ** 2)) + start_angle) * -R * x - R * y

    return is_equation

cx, cy = 0, 0
sx, sy = 10, 10 * width / height
bounds = ((cx-sx/2., cy-sy/2.), (cx+sx/2., cy+sy/2.))

zminmax = 1.0
dist = 0.01


class EQCallback(object):
    def __init__(self):
        self.first_run = True
        self.uv = None
        self.yellow = None
        self.orange = None
        self.red = None
        self.r = self.y = self.o = None
        self.out_array = None

    def equation_callback(self, frame, coords, finished):
        array = frame
        array = array.permute(2, 1, 0)[None, ...]

        x_mult = bounds[1][0] - bounds[0][0]
        y_mult = bounds[1][1] - bounds[0][1]

        # [..., 0] -> convert to black/white
        if self.uv is None:
            self.uv = (
                coords[0][..., 0] / height * x_mult + bounds[0][0],
                coords[1][..., 0] / width * y_mult + bounds[0][1]
            )

        x = eq(self.uv)

        # eq to snap min/mix dist from equation to 0-1 so whole window is red/yellow
        # However, it looked bad for some equations, so changed it to a variable that the user could modify
        x = ((x + zminmax) / (zminmax + zminmax)) * 2 - 1
        x = (x.permute(1, 0)[None, ...]).to(array.device)

        if self.yellow is None:
            self.yellow = (torch.tensor([0.0, 1.0, 1.0])[None, :, None, None]).to(array.device)
            self.orange = (torch.tensor([0.0, 0.5, 1.0])[None, :, None, None]).to(array.device)
            self.red = (torch.tensor([0.0, 0.0, 1.0])[None, :, None, None]).to(array.device)

        if self.r is None:
            self.r = torch.zeros_like(x)  # red
            self.y = torch.zeros_like(x)  # orange
            self.o = torch.zeros_like(x)  # yellow

        x_more = x[...] > 0
        x_less = x[...] < 0

        self.r[...] = 0
        self.o[...] = 0
        self.y[...] = 0

        self.r[...][x_more] = (1.0 - torch.clamp(x[...][x_more], max=1.0))
        self.o[...][x_less] = (1.0 + torch.clamp(x[...][x_less], min=-1.0))
        # original: y[...] = torch.where(torch.abs(x[:, :, ...]) < dist,1.0,0.0)
        self.y[...] = (torch.abs(x[:, :, ...]) < dist).to(self.y.dtype)
        if self.out_array is None:
            self.out_array = torch.zeros_like(array)
        self.out_array = torch.clamp(
            self.r[:, None, ...] * self.red +
            self.o[:, None, ...] * self.orange +
            self.y[:, None, ...] * self.yellow,
            min=0.0, max=1.0)

        array = self.out_array.to(array.dtype)

        array = array.squeeze().permute(2, 1, 0)
        frame[coords] = array[coords]


eqc = EQCallback()
equation_shader = pixel_shader(eqc.equation_callback)



def cv_display_info(arr):
    cw = 12
    ch = 18
    tl = f"{bounds[0][0]},{bounds[0][1]}"
    br = f"{bounds[1][0]},{bounds[1][1]}"
    tr = f"{zminmax}, {dist}"
    arr[...] = cv2.putText(arr, tl, (int(cw), int(ch)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (1.0, 1.0, 1.0))
    arr[...] = cv2.putText(arr, br, (int(width - cw * (len(br)+2)), int(height - ch * .5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
               (1.0, 1.0, 1.0))
    arr[...] = cv2.putText(arr, tr, (int(width - cw * (len(tr) + 2)), int(ch)),
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           (1.0, 1.0, 1.0))

@key_loop
def keys(k):
    global zminmax, dist
    if k == 'q':
        zminmax += 1.0
        equation_shader(img)
        cv_display_info(img)
    elif k == 'a':
        zminmax -= 1
        equation_shader(img)
        cv_display_info(img)
    elif k == 'e':
        dist += 0.01
        equation_shader(img)
        cv_display_info(img)
    elif k == 'd':
        dist -= 0.01
        equation_shader(img)
        cv_display_info(img)


    print(f'zminmax:{zminmax}, dist:{dist}')


equation_shader(img)
cv_display_info(img)
d = display(img, blocking=True)
