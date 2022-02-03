from biocad.core import get_grid, transform, show_all
from biocad.shapes import torus, cardioid_prism

thresh = 1.0

x, y, z, rgb = get_grid()

from biocad.core import transform, show_all
from biocad.shapes import torus, cardioid_prism

obj_list = []

xyz1 = transform(x, y, z, translate=[.5, .5, .5], scale=[.2, .2 + .0625, .25 + .25])
obj_list.append(cardioid_prism(*xyz1, .2, .075))
xyz2 = transform(x, y, z, translate=[.5 + .0625/4.0, .5, .5], scale=[.2, .2 + .0625, .25 + .25])
obj_list.append(torus(*xyz2, .15, .04))

show_all(obj_list, rgb)
