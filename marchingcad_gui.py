from tkinter import *
from tkinter import ttk
from pathlib import Path

from biocad.core import get_grid, transform, show_all

thresh = 1.0

x, y, z, rgb = get_grid()

obj_list = []

root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid(row=0, column=0, columnspan=4, sticky=N + S + W + E)
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

frm_top_buttons = ttk.Frame(frm, padding=10)
frm_top_buttons.grid(column=0, row=0, sticky=N + S + W + E)

t = Text(frm)
t.grid(column=0, row=1, sticky=N + S + W + E)
frm.grid_columnconfigure(0, weight=1)
frm.grid_rowconfigure(1, weight=1)
import torch

from biocad import core
from biocad.shapes import *
from biocad.core import union, intersection, difference
import time
from biocad.stopthread import StoppableThread

thr = None
from math import pi

cached_obs = None

def run_callback(*args):
    print(args)
    global thr, cached_obs

    exec(t.get("1.0", 'end-1c'))
    loc = list(locals().items())

    cached_obs = [l[1] for l in loc if torch.is_tensor(l[1])]
    #todo: cache these so they can be saved immedietely
    if len(cached_obs) > 0:
        if thr is not None:
            '''core.is_closed = True
            thr.join(.1)
            if thr.is_alive():
                del thr
                thr = None'''
            core.geometry_update_data = core.get_showable(cached_obs, rgb)
        else:
            thr = StoppableThread(target=show_all, args=(cached_obs, rgb))
            thr.start()
            core.is_closed = False  # bug fix

    else:
        print("No geometry to show")


def n_popup():
    toplevel = Toplevel()
    label1 = Label(toplevel, text="Enter the new grid size:", height=0, width=100)
    label1.pack()
    label2 = Entry(toplevel,width=100)
    label2.pack()

    def n_change():
        global x, y, z, rgb
        x, y, z, rgb = get_grid(int(label2.get()))
        torch.cuda.empty_cache()
        run_callback()
        toplevel.destroy()

    ok_button = Button(toplevel, text="OK", command=n_change)
    ok_button.pack()

    toplevel.focus_force()


run_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "run.png"))
clear_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "clear.png"))
grid_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "grid.png"))

run_photo = PhotoImage(file=run_png)
clear_photo = PhotoImage(file=clear_png)
grid_photo = PhotoImage(file=grid_png)

Button(frm_top_buttons, text='Compile Object Code', image=run_photo, command=run_callback).pack(side=LEFT)
Button(frm_top_buttons, text='Clear View', image=clear_photo).pack(side=LEFT)
Button(frm_top_buttons, text='Setup Marching Cubes Grid', image=grid_photo, command=n_popup).pack(side=LEFT)

root.bind('<Control-r>', run_callback)
root.bind('<Control-n>', n_popup)

ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=1)
# t.pack()
root.mainloop()
