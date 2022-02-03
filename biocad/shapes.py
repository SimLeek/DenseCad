import torch

# shapes:
#  rounded_rect_prism(x, y, z, rounding_radius, width, height, depth)
#  rect_prism(x, y, z, width, height, depth)
#  torus(x, y, z, inner_radius, tube_radius)
#  sphere(x, y, z, radius)
#  cardioid_prism(x, y, z, a, height)
#  cylinder(x, y, z, radius, height)


def rounded_rect_prism(x, y, z, rounding_radius, width, height, depth):
    a = (torch.abs(x) / width) ** (2 * width / rounding_radius)
    b = (torch.abs(y) / height) ** (2 * height / rounding_radius)
    c = (torch.abs(z) / depth) ** (2 * depth / rounding_radius)
    t = a + b + c
    return t


def rect_prism(x, y, z, width, height, depth):
    a = torch.where(
        torch.abs(x) > width,
        torch.ones_like(x) * torch.finfo(torch.float32).max,
        torch.zeros_like(x)
    )
    b = torch.where(
        torch.abs(y) > height,
        torch.ones_like(x) * torch.finfo(torch.float32).max,
        torch.zeros_like(x)
    )
    c = torch.where(
        torch.abs(z) > depth,
        torch.ones_like(x) * torch.finfo(torch.float32).max,
        torch.zeros_like(x)
    )
    t = a + b + c
    return t


def polygon_prism(x, y, z, boundary_list, depth):
    t = torch.zeros_like(x)
    for boundary in boundary_list:
        if len(boundary)==2:
            a, b = boundary
            c = 0
        else:
            a, b, c = boundary
        t += torch.where(
            x*a+y*b+c > 1.0,
            torch.ones_like(x) * torch.finfo(torch.float32).max,
            torch.zeros_like(x)
        )
    t += torch.where(
        torch.abs(z) > depth,
        torch.ones_like(x) * torch.finfo(torch.float32).max,
        torch.zeros_like(x)
    )

    return t


def torus(x, y, z, inner_radius, tube_radius):
    d0 = (inner_radius - torch.sqrt(x ** 2 + y ** 2))
    d1 = (d0 ** 2 + z ** 2)
    d = d1 / (tube_radius ** 2)
    return d


def sphere(x, y, z, radius):
    d = 1.0 - torch.sqrt(x ** 2 + y ** 2 + z ** 2)/radius
    return d


def cardioid_prism(x, y, z, a, height, b=None):
    if b is None:
        b=a
    c = torch.where(
        torch.abs(z) > height,
        torch.ones_like(x) * torch.finfo(torch.float32).max,
        torch.where(
            (x ** 2 + y ** 2 + a * x) ** 2 > b ** 2 * (x ** 2 + y ** 2),
            torch.ones_like(x) * torch.finfo(torch.float32).max, torch.zeros_like(x)
        )
    )

    return c


def cylinder(x, y, z, radius, height):
    c = torch.where(
        torch.abs(z) > height,
        torch.ones_like(x) * torch.finfo(torch.float32).max,
        torch.where(
            torch.sqrt(x ** 2 + y ** 2) > radius,
            torch.ones_like(x) * torch.finfo(torch.float32).max, torch.zeros_like(x)
        )
    )

    return c
