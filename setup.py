from setuptools import find_packages, setup

setup(
    name="Korean STT toolkit using Tensorflow", 
    version="1.0", 
    author="Hyunsoo Son",
    packages=find_packages(include=("tensorflow_asr", "tensorflow_asr.*"))
)
