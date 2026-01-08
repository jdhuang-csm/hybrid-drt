from setuptools import setup, find_packages


setup(name='hybrid-drt',
      version='0.1',
      description='A Python package for probabilistic electrochemical analysis',
      url='https://github.com/jdhuang-csm/hybrid-drt',
      author='Jake Huang',
      author_email='jdhuang@mines.edu',
      license='BSD 3-clause',
      packages=find_packages(),
      install_requires=[
          'numpy >= 2.0.0',
          'pandas',
          'scipy',
          'cvxopt',
          'matplotlib',
          'scikit-learn',
          'scikit-image'
      ],
      include_package_data=True
      )
