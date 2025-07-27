from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="retinoextract",
    version="1.0.0",
    author="Fareeha Jabeen",
    author_email="fareeha@mcs.nust.edu.pk",
    description="A professional medical image feature extraction toolbox for retinal image analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FareehaJabeen275/RetinoExtractToolbox",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.2.0", "black>=21.0.0", "flake8>=3.9.0", "mypy>=0.910"],
        "docs": ["sphinx>=4.0.0", "sphinx-rtd-theme>=0.5.0"],
    },
    entry_points={
        "console_scripts": [
            "retinoextract=retinoextract.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "retinoextract": ["resources/*", "resources/**/*"],
    },
    keywords="medical imaging, feature extraction, retinal analysis, image processing, computer vision",
    project_urls={
        "Bug Reports": "https://github.com/FareehaJabeen275/RetinoExtractToolbox/issues",
        "Source": "https://github.com/FareehaJabeen275/RetinoExtractToolbox",
        "Documentation": "https://retinoextract.readthedocs.io/",
    },
)
