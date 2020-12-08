from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='inference toolkit',
      license='MIT',
      author='Clément Pinard',
      author_email='clempinard@gmail.com',
      description='Inference and evaluation routines to test on a dataset constructed with validation set constructor',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'depth_evaluation = evaluation_toolkit.depth_evaluation:main'
          ]
      },
      install_requires=[
          'numpy',
          'pandas',
          'path',
          'imageio',
          'scikit-image',
          'scipy',
          'tqdm'
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Science/Research"
      ]
      )
