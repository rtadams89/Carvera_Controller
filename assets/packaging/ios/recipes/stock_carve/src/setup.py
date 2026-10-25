"""Build _stock_carve against the repo's C sources (STOCK_CARVE_SRC)."""

import os

from setuptools import Extension, setup

src = os.environ["STOCK_CARVE_SRC"]
names = ["profile.c", "heightmap.c", "cylindrical.c", "voxel.c", "laser.c", "mesh.c", "_stock_carve.c"]

setup(
    name="stock_carve",
    ext_modules=[
        Extension(
            "_stock_carve",
            sources=[os.path.join(src, name) for name in names],
            include_dirs=[src],
            extra_compile_args=["-std=c99", "-O3"],
            libraries=["m"],
        )
    ],
)
