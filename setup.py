"""Setup file for id8r package."""
from setuptools import setup, find_packages

setup(
    name="id8r",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pytest>=7.0.0",
        "pytest-mock>=3.10.0",
    ],
    python_requires=">=3.9",
)
