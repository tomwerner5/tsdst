import os
import codecs

from setuptools import setup, find_packages


with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()
    
def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")
    
setup(
    name='tsdst',
    version=get_version("tsdst/__init__.py"),
    description='A low-key data science and statistics toolkit',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Tom W',
    author_email='tomwerner5@gmail.com',
    url='https://tomwerner5.github.io/tsdst/',
    download_url='https://github.com/tomwerner5/tsdst/archive/v_' + get_version("tsdst/__init__.py") + 'tar.gz',
    packages=find_packages(include=['tsdst', 'tsdst.nn']),
    keywords=['data science', 'statistics', 'neural network', 'bayesian'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    install_requires=['matplotlib>=3.1.0',
                      'numba>=0.44.1',
		      'numdifftools>=0.9.39',
		      'numpy>=1.16.4',
                      'pandas>=0.24.2',
                      'scikit-learn>=0.22.1',
                      'scipy>=1.2.1',
		      'seaborn>=0.11.0',
                      'sqlalchemy>=1.3.4',
                      'statsmodels>=0.9.0',
                      ]
)
    
