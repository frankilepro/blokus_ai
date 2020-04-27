# blokus-ai

[![Blokus Gym Environment](https://github.com/frankilepro/blokus_ai/workflows/Blokus%20Gym%20Environment/badge.svg)](https://github.com/frankilepro/blokus_ai/actions?query=workflow%3A%22Blokus+Gym+Environment%22)
[![Upload Blokus Gym Env to PyPi](https://github.com/frankilepro/blokus_ai/workflows/Upload%20Blokus%20Gym%20Env%20to%20PyPi/badge.svg)](https://github.com/frankilepro/blokus_ai/actions?query=workflow%3A%22Upload+Blokus+Gym+Env+to+PyPi%22)

## Install project locally

This will compile the project using Cython, you need to run this every time you make changes to the code

```bash
pip install blokus-gym
```

If you wish to debug or simply not to re-compile at every operation, you can do this:

```bash
git add . && git commit -m "commit what is not the .c files"
cd blokus && git clean -fdx && cd ..
```

## Add a new environment

You can easily add new configurations by adding another class in the [blokus_gym/envs/blokus_envs.py](blokus_gym/envs/blokus_envs.py) file following the current structure:

```python
class BlokusCustomEnv(BlokusEnv):
    NUMBER_OF_PLAYERS = 3
    BOARD_SIZE = 10  # This will result in a 10x10 board
    STATES_FILE = "states_custom.json"  # This needs to be set, if not it will take the base class states
    all_shapes = [shape for shape in get_all_shapes()
                  if shape.size == 4]  # This will take only the 4 tiles pieces
    bot_type = CustomPlayer  # Defaults to RandomPlayer if not passed
```

Then you need to add your new Env to the list in [blokus_gym/envs/\_\_init\_\_.py](blokus_gym/envs/__init__.py) as well as register your environment in [blokus_gym/\_\_init\_\_.py](blokus_gym/__init__.py)

### Add a new player

You can inspire yourself from the other players like [blokus_gym/envs/players/greedy_player.py](blokus_gym/envs/players/greedy_player.py)

```python
class CustomPlayer(Player):
    def do_move(self):
        return self.sample_move()
```
