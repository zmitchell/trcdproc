import io
import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    readme = '\n' + f.read()

# with open('README.md') as readme_file:
#     readme = readme_file.read()

requirements = [
    'h5py>=2.7.0',
    'numpy>=1.13.1',
    'scipy>=0.19.1',
]

test_requirements = [
    'pytest',
]

setup(
    name='trcdproc',
    version='0.0.0',
    description="This is a library for handling and processing TRCD data stored in HDF5 files.",
    long_description=readme + '\n\n',
    author="Zach Mitchell",
    author_email='zmitchell@fastmail.com',
    url='https://github.com/zmitchell/trcdproc',
    packages=find_packages(exclude='tests',),
    package_dir={'trcdproc':
                 'trcdproc'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='trcdproc',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
    ],
    test_suite='tests',
    tests_require=test_requirements,
    entry_points={
        'console_scripts': [
            'trcdproc = trcdproc.__main__:main'
        ]
    }
)
