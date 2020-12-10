import os
import codecs


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open("README.md", 'r') as f:
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
    description='A library of convenience functions for Tom',
    long_description=long_description,
    author='Tom W',
    author_email='tomwerner5@gmail.com',
    url="https://www.tmwerner.com",
    packages=['tsdst'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Mathmatics :: Statistics'
    ],
    install_requires=['numpy>=1.16.4',
                      'pandas>=0.24.2',
                      'scikit-learn>=0.22.1',
                      'statsmodels>=0.9.0',
                      'matplotlib>=3.1.0',
                      'sqlalchemy>=1.3.4',
                      'scipy>=1.2.1',
                      'numba>=0.44.1',
		      'seaborn>=0.11.0',
                      ]
)
    