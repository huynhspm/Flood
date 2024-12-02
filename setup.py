#!/usr/bin/env python

from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split("\n")

setup(
    name="Flood",
    version="0.1",
    description="Project to predict water level",
    author="huynhspm",
    author_email="huynhngoctrinh542002@gmail.com",
    url="https://github.com/huynhspm/Flood",
    packages=find_packages(where="src"),  # Tìm các package trong thư mục src
    package_dir={"": "src"},  # Thư mục gốc chứa mã nguồn là src
    entry_points={
        "console_scripts": [
            "train_command = Flood.train:main",  # Giả sử bạn có file train.py trong Flood
            "eval_command = Flood.eval:main",  # Giả sử bạn có file eval.py trong Flood
        ]
    },
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=requirements,
)
