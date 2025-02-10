"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages


setup(
    name="movement_disorder_deep_learning",  # Required
    version="0.0.1",  # Required
    description="Movement disorder deep learning",  # Optional
    author="Anthony Lee",  # Optional
    author_email="anthony8lee@gmail.com",  # Optional
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.7, <4",
)