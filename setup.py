import setuptools

with open('README.md') as fp:
    long_description = fp.read()

with open('requirements.txt') as fp:
    requirements = fp.read().splitlines()

setuptools.setup(
    name='torchkan',
    version='0.18.3',
    description='KAN Module',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ivan Drokin',
    maintainer='jshn9515',
    maintainer_email='jshn9510@gmail.com',
    license='MIT',
    keywords=['computer-vision', 'convolutional-neural-networks',
              'kolmogorov-arnold-networks'],
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
