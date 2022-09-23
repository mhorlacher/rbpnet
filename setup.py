# %%
from setuptools import setup, find_packages

# %%
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# %%
setup(name='rbpnet',
      version='0.10.0',
      description='RBPNet',
      url='http://github.com/mhorlacher/rbpnet',
      author='Marc Horlacher',
      author_email='marc.horlacher@gmail.com',
      license='MIT',
      install_requires=requirements,
      packages=find_packages(),
      include_package_data=True,
      # data_files = [
      #       #('notebooks', ['rbpnet/notebooks/Evaluate.ipynb']),
      #       ('configs', ['rbpnet/configs/config.modisco.gin'])
      #       ],
      #package_data={'notebooks': ['notebooks/*.ipynb']},
      entry_points = {
            'console_scripts': [
                  'rbpnet=rbpnet.__main__:main',
            ],
      },
      zip_safe=False)