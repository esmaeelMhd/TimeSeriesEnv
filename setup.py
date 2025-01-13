from setuptools import setup, find_packages

setup(
    name='TimeSeriesEnv',
    version='1.0.0',
    author='Esmaeel Mohammadi',
    author_email='esm@kruger.dk, esmo@bio.aau.dk, s.mohammadi.rl@gmail.com',
    description='A customizable and scalable simulation environment for time series data analysis and training of reinforcement learning algorithms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/esmaeelMhd/TimeSeriesEnv',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'gym',
        'numpy',
        'pandas',
        'plotly',
        'torch',
        # Add other dependencies as required
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial 4.0 International Public License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
