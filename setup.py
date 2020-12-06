from setuptools import find_packages, setup

setup(
    name='a2c-ppo-acktr',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['matplotlib', 'pybullet'])
    # Removed gym as a dependency since we're using a local version here
