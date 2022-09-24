from setuptools import setup, find_packages

# with open("README.md", "r",encoding="utf8") as fh:
#     long_description = fh.read()

setup(name='hybrid-drt',
      version='0.1',
      description='A Python package for analyzing square wave impedance data',
      # long_description=long_description,
      # long_description_content_type="text/markdown",
      # url='https://github.com/jdhuang-csm/bayes-drt',
      author='Jake Huang',
      author_email='jdhuang@mines.edu',
      license='BSD 3-clause',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'cvxopt',
          # 'cmdstanpy',
          'matplotlib',
          'mitlef',
          'pytorch',
          'gpytorch'
      ],
      include_package_data=True
      )
