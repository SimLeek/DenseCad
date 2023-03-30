import tempfile
from tkinter import Tk, ttk, N, S, W, E, Text, PhotoImage, Button, LEFT
from pathlib import Path
import open3d as o3d
import zipfile

from biocad import core
from biocad.stopthread import StoppableThread
from biocad.core import show_all
from biocad.assembly import Showable
from typing import List
import numpy as np
from scipy.spatial.transform import Rotation as R

def show_assembly(drawables):
    o3d.visualization.draw_geometries_with_animation_callback(drawables, core.func,
                                                              window_name='Marching cubes',
                                                              width=int(1920 // 2), height=1080,
                                                              left=int(1920 // 2), top=0)


def get_run_callback(t, thr, cached_obs):
    def run_callback():
        nonlocal t, thr, cached_obs

        from biocad.assembly import load
        from biocad.core import place_list_along_parametric_curve, copy_along_parametric_curve
        from numpy import sin, cos, tan, pi

        text = t.get("1.0", 'end-1c')
        exec(text)
        loc = list(locals().items())

        loc2 = []
        for l in loc:
            if isinstance(l[1], (tuple, list)):
                for e, li in enumerate(l[1]):
                    loc2.append((l[0]+str(e), li))
            else:
                loc2.append(l)
        cached_obs = [l for l in loc2 if isinstance(l[1], Showable)]
        cached_obs = [c[1] for c in cached_obs]

        drawables = []
        for s in cached_obs:
            drawables.append(s.get_showable())

        if len(cached_obs) > 0:
            if not isinstance(thr, list):
                core.geometry_update_data = drawables
            else:

                thr = StoppableThread(target=show_assembly, args=(drawables,))
                thr.start()
                core.is_closed = False  # bug fix

        else:
            print("No geometry to show")

    return run_callback


def run_assembly_gui(file=None):
    root = Tk()
    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, columnspan=4, sticky=N + S + W + E)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    frm_top_buttons = ttk.Frame(frm, padding=10)
    frm_top_buttons.grid(column=0, row=0, sticky=N + S + W + E)

    t = Text(frm, undo=True)
    t.grid(column=0, row=1, sticky=N + S + W + E)
    frm.grid_columnconfigure(0, weight=1)
    frm.grid_rowconfigure(1, weight=1)

    thr = []
    cached_obs = []

    run_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "run.png"))
    export_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "export.png"))
    import_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "import.png"))
    save_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "save.png"))
    open_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "load.png"))

    run_photo = PhotoImage(file=run_png)
    export_photo = PhotoImage(file=export_png)
    import_photo = PhotoImage(file=export_png)
    save_photo = PhotoImage(file=save_png)
    open_photo = PhotoImage(file=open_png)

    # python files in text and transforms on the variables

    run_callback = get_run_callback(t, thr, cached_obs)

    Button(frm_top_buttons, text='Compile Assembly Code', image=run_photo, command=run_callback).pack(side=LEFT)

    root.bind('<Control-r>', run_callback)

    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=1)

    root.mainloop()

if __name__ == '__main__':
    run_assembly_gui()