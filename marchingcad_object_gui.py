import pathlib
import zipfile
from tkinter import *
from tkinter import ttk
from pathlib import Path
from tkinter import filedialog, messagebox
from biocad.core import get_grid, transform, show_all
import open3d as o3d
import tempfile
import shutil
from biocad import core
from biocad.stopthread import StoppableThread
import torch


def exec_make_shapes(t, x, y, z):
    from biocad.shapes import (
        rounded_rect_prism,
        rect_prism,
        polygon_prism,
        torus,
        sphere,
        cardioid_prism,
        cylinder
    )
    from biocad.core import get_grid, transform, show_all
    from biocad.core import union, intersection, difference
    from math import pi

    exec(t.get("1.0", 'end-1c'))
    loc = list(locals().items())

    cached_obs = [l for l in loc if torch.is_tensor(l[1])]
    cached_obs = [c[1] for c in cached_obs if c[0] not in ['x', 'y', 'z', 'rgb']]

    return cached_obs


def get_run_callback(t, thr, cached_obs, x, y, z, rgb):
    def run_callback(*args):

        nonlocal t, thr, cached_obs, x, y, z, rgb

        print(args)

        cached_obs = exec_make_shapes(t, x, y, z)
        # todo: cache these so they can be saved immedietely
        if len(cached_obs) > 0:
            if not isinstance(thr, list):
                core.geometry_update_data = core.get_showable(cached_obs, rgb)
            else:
                thr = StoppableThread(target=show_all, args=(cached_obs, rgb))
                thr.start()
                core.is_closed = False  # bug fix

        else:
            print("No geometry to show")

    return run_callback


def get_n_popup(t, thr, cached_objs, run_cb, x, y, z, rgb):
    def n_popup():
        nonlocal t, thr, cached_objs
        toplevel = Toplevel()
        label1 = Label(toplevel, text="Enter the new grid size:", height=0, width=100)
        label1.pack()
        label2 = Entry(toplevel, width=100)
        label2.pack()

        def n_change():
            nonlocal x, y, z, rgb
            x, y, z, rgb = get_grid(int(label2.get()))
            torch.cuda.empty_cache()
            run_cb(t, thr, cached_objs)
            toplevel.destroy()

        ok_button = Button(toplevel, text="OK", command=n_change)
        ok_button.pack()

        toplevel.focus_force()

    return n_popup


def get_export_callback(t, cached_obs, x, y, z, rgb):
    def export_callback():
        nonlocal t, cached_obs, x, y, z, rgb
        if cached_obs is None or len(cached_obs) == 0:
            try:
                cached_obs = exec_make_shapes(t, x, y, z)
            except Exception as e:
                messagebox.showerror("ERROR", "No objects are available to export. "
                                              "generating current code gave error: " + str(e))
                return
            if len(cached_obs) == 0:
                messagebox.showerror("ERROR", "No objects are available to export. "
                                              "Code is error free, but generated no objects.")
                return

        f = filedialog.asksaveasfile(
            mode='w',
            defaultextension='.stl',
            filetypes=[
                ('StereoLithography', '*.stl'),
                ('Object File', '*.obj'),
                ('Polygon File Format', '*.ply'),
                ('Object File Format', '*.off'),
                ('GL Transmission Format', '*.gltf')
            ]
        )
        if f is None:  # dialog was closed with cancel
            return

        ext = pathlib.Path(f.name).suffix

        mesh = core.get_showable(cached_obs, rgb)[0]
        if ext == '.stl':
            norm_mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
            o3d.io.write_triangle_mesh(f.name, norm_mesh)
        else:
            o3d.io.write_triangle_mesh(f.name, mesh)

        messagebox.showinfo("Success", "Objects successfully exported!")

    return export_callback


def get_save_callback(t, rgb):
    def save_callback():
        nonlocal t, rgb

        f = filedialog.asksaveasfile(
            mode='w',
            defaultextension='.zmarch',
            filetypes=[
                ('Marching Cubes Code', '*.march'),
                ('Zipped Marching Cubes Code and PLY', '*.zmarch'),
            ]
        )

        if f is None:  # dialog was closed with cancel
            return

        ext = pathlib.Path(f.name).suffix

        text = t.get("1.0", 'end-1c')

        if ext == '.march':
            f.write(text)
            f.close()
        elif ext == '.zmarch':
            fname = f.name
            f.close()
            stem = pathlib.Path(f.name).stem
            march = stem + '.march'
            stl = stem + '.ply'

            with tempfile.TemporaryDirectory() as tmpdirname:
                try:
                    exec(text)
                    loc = list(locals().items())
                    cached_obs = [l[1] for l in loc if torch.is_tensor(l[1])]
                except Exception as e:
                    messagebox.showerror("WARNING", "No objects are available to export. "
                                                    "generating current code gave error: " + str(e))
                if len(cached_obs) == 0:
                    messagebox.showerror("WARNING", "No objects are available to export. "
                                                    "Code is error free, but generated no objects.")

                full_stl = pathlib.Path(tmpdirname, stl)
                mesh = core.get_showable(cached_obs, rgb)[0]
                o3d.io.write_triangle_mesh(str(full_stl), mesh)

                # todo: modify text to contain the warnings or checksum if generation succeeded
                full_march = pathlib.Path(tmpdirname, march)
                with open(full_march, mode='w') as f_march:
                    f_march.write(text)

                shutil.make_archive(stem, 'zip', tmpdirname)
                shutil.move(stem + '.zip', fname)
        else:
            messagebox.showerror("ERROR", f'"{ext}" is not a supported file type.')

    return save_callback


def get_open_callback(text):
    def open_callback(file=None):
        nonlocal text

        if file is None:
            f = filedialog.askopenfile(
                mode='r',
                defaultextension='.zmarch',
                filetypes=[
                    ('Marching Cubes Code', '*.march'),
                    ('Zipped Marching Cubes Code and PLY', '*.zmarch'),
                    ('All', '*.*')
                ])
        else:
            f = file

        if f is None:  # dialog was closed with cancel
            return

        pname = pathlib.Path(f.name)
        ext = pname.suffix
        stem = pname.stem

        if ext == '.zmarch':
            archive = zipfile.ZipFile(f.name, 'r')
            march_file = archive.open(stem + '.march')
            text = march_file.read()
            march_file.close()
            archive.close()
        else:
            text = f.read()
        text.delete("1.0", 'end-1c')
        text.insert("1.0", text)

    return open_callback


def run_object_gui(file=None):
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

    t = Text(frm, undo=True)
    t.grid(column=0, row=1, sticky=N + S + W + E)
    frm.grid_columnconfigure(0, weight=1)
    frm.grid_rowconfigure(1, weight=1)

    thr = []

    cached_obs = []

    run_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "run.png"))
    clear_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "clear.png"))
    grid_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "grid.png"))
    export_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "export.png"))
    save_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "save.png"))
    open_png = str(Path.joinpath(Path(__file__).parent, "gui_icons", "load.png"))

    run_photo = PhotoImage(file=run_png)
    clear_photo = PhotoImage(file=clear_png)
    grid_photo = PhotoImage(file=grid_png)
    export_photo = PhotoImage(file=export_png)
    save_photo = PhotoImage(file=save_png)
    open_photo = PhotoImage(file=open_png)

    run_callback = get_run_callback(t, thr, cached_obs, x, y, z, rgb)
    n_popup = get_n_popup(t, thr, cached_obs, run_callback, x, y, z, rgb)
    export_callback = get_export_callback(t, cached_obs, x, y, z, rgb)
    save_callback = get_save_callback(t, rgb)
    open_callback = get_open_callback(t)

    if file is not None:
        open_callback(file=file)

    Button(frm_top_buttons, text='Compile Object Code', image=run_photo, command=run_callback).pack(side=LEFT)
    Button(frm_top_buttons, text='Clear View', image=clear_photo).pack(side=LEFT)
    Button(frm_top_buttons, text='Setup Marching Cubes Grid', image=grid_photo, command=n_popup).pack(side=LEFT)
    Button(frm_top_buttons, text='Export Generated Object', image=export_photo, command=export_callback).pack(side=LEFT)
    Button(frm_top_buttons, text='Save Object Code', image=save_photo, command=save_callback).pack(side=LEFT)
    Button(frm_top_buttons, text='Open Object Code', image=open_photo, command=open_callback).pack(side=LEFT)

    root.bind('<Control-r>', run_callback)
    root.bind('<Control-n>', n_popup)
    root.bind('<Control-e>', export_callback)
    root.bind('<Control-s>', save_callback)
    root.bind('<Control-s>', open_callback)

    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=1)
    # t.pack()
    root.mainloop()


if __name__ == '__main__':
    run_object_gui()
