from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="TimeSeriesEnv",
    version="1.0.0",
    author="Esmaeel Mohammadi",
    author_email="esm@kruger.dk, esmo@bio.aau.dk, s.mohammadi.rl@gmail.com",
    description="A Gymnasium environment for reinforcement learning on time-series control problems.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/esmaeelMhd/TimeSeriesEnv",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29",
        "joblib>=1.3",
        "matplotlib>=3.7",
        "numpy>=1.24",
        "pandas>=2.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.3",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "dev": ["pytest>=8.0"],
        "all": ["torch>=2.0", "pytest>=8.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
