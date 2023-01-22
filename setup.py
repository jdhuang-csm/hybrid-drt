from setuptools import setup, find_packages

# with open("README.md", "r",encoding="utf8") as fh:
#     long_description = fh.read()

setup(name='hybrid-drt',
      version='0.1',
      description='A Python package for probabilistic electrochemical analysis',
      # long_description=long_description,
      # long_description_content_type="text/markdown",
      url='https://github.com/jdhuang-csm/hybrid-drt',
      author='Jake Huang',
      author_email='jdhuang@mines.edu',
      license='BSD 3-clause',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'cvxopt',
          'matplotlib',
          'scikit-learn',
      ],
      include_package_data=True
      )
