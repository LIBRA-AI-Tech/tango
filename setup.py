#!/usr/bin/env python

import setuptools

setuptools.setup(
    name='cpp_solver',
    version='0.0.1',
    description='A complete Coverage Path Planning solver based on AIPlan4EU Unified Planning library.',
    packages=setuptools.find_packages(),
    install_requires=[
        "unified-planning[engines]",
        "jinja2>=3.1.2,<3.2.0",
        "Pillow>=9.5.0,<9.6.0",
        "pygeos>=0.14,<0.15",
        "scikit-image>=0.21.0,<0.22.0",
        "opencv-python>=4.7.0.72,<4.8.0.00",
        "matplotlib>=3.7.1,<3.8.0",
    ],
    python_requires='>=3.9',
    zip_safe=False,
)
