from setuptools import find_packages, setup

requirements = [
    'matplotlib',
    'numpy',
    'pandas',
    'tqdm'
]

version = 0.1

setup(
    name='ml',
    version=version,
    python_requires='>=3.6',
    author='Matheus Xavier',
    author_email='xavier_sampaio@alu.ufc.br',
    url='https://github.com/flych3r/class-machine-learning',
    packages=[
        'ml',
        'ml.core',
    ],
    # packages=find_packages(),
    description='ML algorithms',
    license='MIT-0',
    zip_safe=True,
    install_requires=requirements,
)