"""kivy-ios recipe: static libstock_carve.a plus an importable _stock_carve module."""

import os

from kivy_ios.toolchain import CythonRecipe

# kivy-ios chdirs into earlier recipes' build dirs before this recipe is
# downloaded, and --add-custom-recipe is passed as a relative path. Freeze
# absolute paths at import, while cwd is still the repo root.
_RECIPE_DIR = os.path.dirname(os.path.abspath(__file__))
_NATIVE_SRC = os.path.abspath(
    os.path.join(
        _RECIPE_DIR,
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


class StockCarveRecipe(CythonRecipe):
    version = "1.0.0"
    # Local setup.py. Resolved against recipe_dir; a relative recipe_dir makes
    # kivy-ios treat this as a URL and fail with "unknown url type: 'src'".
    url = "src"
    library = "libstock_carve.a"
    depends = ["python3"]
    cythonize = False
    hostpython_prerequisites = []
    call_hostpython_via_targetpython = False

    def init_after_import(self, ctx):
        self.recipe_dir = _RECIPE_DIR

    def get_recipe_env(self, plat):
        env = super().get_recipe_env(plat)
        env["STOCK_CARVE_SRC"] = _NATIVE_SRC
        return env

    def install(self):
        self.install_python_package(name=self.so_filename("_stock_carve"), is_dir=False)


recipe = StockCarveRecipe()
