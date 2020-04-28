import os
import pkg_resources
from setuptools import setup, Extension, find_packages
from Cython.Build import build_ext, cythonize


ext_modules = [
    Extension("blokus_gym.envs.blokus_env", ["blokus_gym/envs/blokus_env.py"]),
    Extension("blokus_gym.envs.blokus_envs", ["blokus_gym/envs/blokus_envs.py"]),
    Extension("blokus_gym.envs.shapes.shape", ["blokus_gym/envs/shapes/shape.py"]),
    Extension("blokus_gym.envs.shapes.shapes", ["blokus_gym/envs/shapes/shapes.py"]),
    Extension("blokus_gym.envs.players.player", ["blokus_gym/envs/players/player.py"]),
    Extension("blokus_gym.envs.players.random_player", ["blokus_gym/envs/players/random_player.py"]),
    Extension("blokus_gym.envs.players.greedy_player", ["blokus_gym/envs/players/greedy_player.py"]),
    Extension("blokus_gym.envs.players.minimax_player", ["blokus_gym/envs/players/minimax_player.py"]),
    Extension("blokus_gym.envs.game.blokus_game", ["blokus_gym/envs/game/blokus_game.py"]),
    Extension("blokus_gym.envs.game.board", ["blokus_gym/envs/game/board.py"]),
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

if 'BLOKUS_VERSION' in os.environ:
    version = os.environ['BLOKUS_VERSION'].split("/")[-1]  # i.e.: v0.21
    cython_params = {}
else:
    try:
        version = pkg_resources.get_distribution("blokus-gym").version
    except Exception:
        version = 'v1.0'
    cython_params = {
        'ext_modules': cythonize(ext_modules),
        'build_ext': build_ext
    }

setup(
    name='blokus-gym',
    packages=find_packages(),
    version=version,
    license='gpl-3.0',
    description='OpenAI gym environment for Blokus',
    url='https://github.com/frankilepro/blokus_ai',
    download_url=f'https://github.com/frankilepro/blokus-ai/archive/{version}.tar.gz',
    keywords=['blokus', 'board game', 'block us'],
    install_requires=[
        'gym',
        'numpy',
        'matplotlib',
        'torch',
        'Cython'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
    ],
    **cython_params
)
