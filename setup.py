from setuptools import setup, find_packages

setup(
    name='SGPocket',
    version='1.1.0',
    description='A Graph Convolutional Neural Network to predict ligand binding site on a protein.',
    author='Kevin Crampon',
    author_email='kevin.crampon@univ-reims.fr',
    url='https://github.com/KevinCrp/SGPocket',
    packages=find_packages(include=["SGPocket",
                                    "SGPocket.utilities",
                                    "SGPocket.networks"]),
    install_requires=[],
)