from setuptools import setup

setup(name='deep_attribution',
      version='1.0',
      description='Attribution Framework',
      author='Leopold Davezac',
      author_email='leopoldavezac@gmail.com',
      url='https://github.com/leopoldavezac/DeepAttribution',
      packages=[
            "deep_attribution.feature_engineering",
            "deep_attribution.preprocess",
            "deep_attribution.train",
            "deep_attribution.model"
            ],
     )