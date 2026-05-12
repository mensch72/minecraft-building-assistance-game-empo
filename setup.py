from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExtWithNumpy(build_ext):
    def finalize_options(self):
        super().finalize_options()

        import numpy

        for extension in self.extensions:
            extension.include_dirs.append(numpy.get_include())

setup(
    cmdclass={"build_ext": BuildExtWithNumpy},
    ext_modules=[
        Extension(
            "_mbag",
            sources=[
                "mbag/c_extensions/_mbagmodule.c",
                "mbag/c_extensions/action_distributions.c",
                "mbag/c_extensions/blocks.c",
                "mbag/c_extensions/mcts.c",
            ],
            include_dirs=[],
        )
    ],
)
