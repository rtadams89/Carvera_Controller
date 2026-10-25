"""python-for-android recipe for the stock-carving C extension."""

import os
from os.path import abspath, dirname, exists, join

from pythonforandroid.recipe import CompiledComponentsPythonRecipe

_SETUP = """
import os
from setuptools import Extension, setup

src = os.environ["STOCK_CARVE_SRC"]
names = ["profile.c", "heightmap.c", "cylindrical.c", "voxel.c", "laser.c", "mesh.c", "_stock_carve.c"]
setup(
    name="stockcarve",
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
"""


class StockCarveRecipe(CompiledComponentsPythonRecipe):
    version = "1.0.0"
    url = None
    name = "stockcarve"
    depends = ["python3"]
    call_hostpython_via_targetpython = False
    # The .so is a top-level module, not a stockcarve package directory.
    site_packages_name = "_stock_carve"

    def _sources(self) -> str:
        return abspath(
            join(
                dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "..",
                "carveracontroller",
                "addons",
                "stock",
                "simulator",
                "native",
            )
        )

    def get_recipe_env(self, arch=None, with_flags_in_cc=True):
        env = super().get_recipe_env(arch, with_flags_in_cc)
        env["CFLAGS"] += " -std=c99 -O3"
        env["STOCK_CARVE_SRC"] = self._sources()
        return env

    def prepare_build_dir(self, arch):
        # python-for-android passes the arch name ("arm64-v8a"), not an Arch.
        build_dir = self.get_build_dir(arch)
        if not exists(build_dir):
            os.makedirs(build_dir)
        setup_path = join(build_dir, "setup.py")
        with open(setup_path, "w", encoding="utf-8") as handle:
            handle.write(_SETUP)


recipe = StockCarveRecipe()
