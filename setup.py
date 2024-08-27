from setuptools import setup, find_packages

setup(
    name="time_series_env",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "gym",
        "numpy",
        "pandas",
    ],
    entry_points={
        'console_scripts': [
            'time_series_env=time_series_env.__main__:main',
        ],
    },
)
