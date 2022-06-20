#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

plot_requirements = []

test_requirements = [
    *plot_requirements,
    "black>=22.3.0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "isort>=5.7.0",
    "mypy>=0.790",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
    "s3fs>=2022.5.0",
    "tox>=3.15.2",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "jupyterlab>=3.2.8",
    "m2r2>=0.2.7",
    "napari[qt]>=0.4.16",
    "Sphinx>=3.4.3",
    "furo>=2022.4.7",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    "aicsimageio~=4.8",
    "dataclasses-json~=0.5",
    "numpy~=1.0",
    "pandas~=1.4",
    "scikit-image~=0.19",
    "torch~=1.11",
    "pytorch-lightning~=1.6.4",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "plot": plot_requirements,
    "all": [
        *requirements,
        *dev_requirements,
        *plot_requirements,
    ],
}

setup(
    author="Gregory Johnson, Eva Maxfield Brown",
    author_email="evamaxfieldbrown@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description=("TODO"),
    entry_points={
        "console_scripts": [
            (),
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="TODO",
    name="label-free",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.8",
    setup_requires=setup_requirements,
    test_suite="label_free/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/open-label-free/label-free",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.0.0",
    zip_safe=False,
)
