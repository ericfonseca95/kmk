# setup.py file for the package kmodels. prerequisite packages are pytorch with cuda enabled, numpy, scipy, scikit-learn, and matplotlib.
from setuptools import setup, find_packages
# Authors : Kalyn Kearney and Eric Fonseca
setup(
    name="kmodels",
    version="0.1.0",
    author="Kalyn Kearney and Eric Fonseca",
    author_email="kalynkearney@ufl.edu", 
    description="A package for ML forces from EMG data",
    license='BSD 2-clause',
    packages=['kmodels'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'pymatgen', 'pandas'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
