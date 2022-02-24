from setuptools import find_packages, setup
from typing import List

setup(
    name="STT project", 
    version="1.0", 
    packages=find_packages(include=("tensorflow_asr", "tensorflow_asr.*")))