from setuptools import setup, find_packages

setup(
    name="dronevla",
    version="0.1.0",
    author="ZHL-max",
    description="Vision-Language-Action Model for Drone Control",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "transformers>=4.30.0",
        "gymnasium>=0.29.0",
        "pybullet>=3.2.5",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
