from setuptools import setup, find_packages

setup(
    name="hsi_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'ipywidgets',
        'ipympl',
        'ipython',
        'h5py',
    ],
    author="Julien Rehault",
    description="A package for HSI data analysis and visualization",
)