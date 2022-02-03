# seperate project

from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
print(graph.get_input_devices())

import cv2

cv2.CAP_DSHOW