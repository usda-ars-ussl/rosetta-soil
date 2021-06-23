import os
from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(HERE, "rosetta", "__version__.py")) as f:
    exec(f.read(), version)

LONG = (
    "Implementation of the Rosetta pedotransfer function model. "
    "Rosetta predicts parameters for the van Genuchten unsaturated soil "
    "hydraulic properties model, using as input basic soil "
    "characterization data such bulk density and percentages of sand, "
    "silt, and clay."
)

SHORT = "Predict soil hydraulic parameters from basic characterization data."

setup(
    name="rosetta-soil",
    version=version["__version__"],
    description=SHORT,
    long_description=LONG,
    url="https://github.com/usda-ars-ussl/rosetta-soil",
    packages=["rosetta"],
    package_data={"rosetta": ["sqlite/Rosetta.sqlite"]},
    python_requires=">=3.7",
    install_requires=["numpy"],
    zip_safe=False,
    license="License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
    author="Todd Skaggs",
    author_email="todd.skaggs@usda.gov",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
