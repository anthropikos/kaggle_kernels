from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
	name = "UNET_GAN",
	version = "0.1",
    author = "Anthony Lee",
    author_email = "anthony8lee@gmail.com",
	packages = find_packages(where="src"),
	package_dir = {"": "src"},
	py_modules = [splitext(basename(path))[0] for path in glob("src/*.py")],
)
