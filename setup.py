from setuptools import setup, find_packages

setup(
    name='DemoRe',
    version='0.2',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='DemoRe',
    long_description=open('README.md').read(),
    install_requires=['numpy'],
    url='https://github.com/RicardoL1u/DemoRe',
    author='RicardoL1u',
    author_email='ricardoliu@outlook.com'
)
