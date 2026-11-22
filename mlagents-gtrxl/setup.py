from setuptools import find_packages, setup


setup(
    name="pathsearch-mlagents-gtrxl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["mlagents==1.1.0", "torch==2.8.0"],
    entry_points={
        "mlagents.trainer_type": [
            "pathsearch_gtrxl=pathsearch_gtrxl.plugin:get_type_and_settings"
        ]
    },
)
