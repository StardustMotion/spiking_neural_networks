from setuptools import setup

setup(name='lifq',
      version='0.1',
      description='Leaky integrate and fire quantizer',
      url='http://github.com/aloaah/lifq',
      author='Aloïs Turuani',
      author_email='alois.turuani@gmail.com',
      license='MIT',
      packages=['lifq'],
      install_requires=[
          'brian2', 'numpy', 'scikit-image'
      ],
      zip_safe=False)