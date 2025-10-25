from setuptools import setup, find_packages

setup(
    name="isac",
    version="0.1.0",
    author="Hunter2030ZeRo",
    author_email="minchankim1127@gmail.com",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.57.0",
        "torch>=2.7.0",
    ],
    python_requires='>=3.10'
)