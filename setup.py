from distutils.core import setup
# from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize, build_ext

# ext_modules = [
#     Extension("blokus.envs",  ["blokus/envs/blokus_env.py"]),
#     Extension("blokus.envs.shapes",  ["blokus/envs/shapes/shapes.py", "blokus/envs/shapes/shape.py"]),
#     Extension("blokus.envs.players",  ["blokus/envs/players/player.py", "blokus/envs/players/random_player.py"]),
#     Extension("blokus.envs.game",  ["blokus/envs/game/blokus_game.py",
#                                     "blokus/envs/game/game.py", "blokus/envs/game/board.py"]),
# ]
ext_modules = [
    Extension("blokus.envs.blokus_env",  ["blokus/envs/blokus_env.py"]),
    Extension("blokus.envs.shapes.shapes", ["blokus/envs/shapes/shapes.py"]),
    Extension("blokus.envs.shapes.shape", ["blokus/envs/shapes/shape.py"]),
    Extension("blokus.envs.players.player", ["blokus/envs/players/player.py"]),
    Extension("blokus.envs.players.random_player", ["blokus/envs/players/random_player.py"]),
    Extension("blokus.envs.game.blokus_game", ["blokus/envs/game/blokus_game.py"]),
    Extension("blokus.envs.game.game", ["blokus/envs/game/game.py"]),
    Extension("blokus.envs.game.board", ["blokus/envs/game/board.py"]),
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
    name='blokus',
    packages=['blokus'],
    version='0.12',
    license='gpl-3.0',
    description='OpenAI gym environment for Blokus',
    url='https://github.com/frankilepro/blokus-ai',
    download_url='https://github.com/frankilepro/blokus-ai/archive/v0.12.tar.gz',
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
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
