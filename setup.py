from distutils.core import setup
# from setuptools import setup, find_packages

setup(
    name='blokus',
    packages=['blokus'],
    version='0.1',
    license='gpl-3.0',
    description='OpenAI gym environment for Blokus',
    url='https://github.com/frankilepro/blokus-ai',
    download_url='https://github.com/frankilepro/blokus-ai/archive/v0.1.tar.gz',
    keywords=['blokus', 'board game', 'block us'],
    install_requires=[
        'gym',
        'numpy',
        'matplotlib'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
    ]
)
