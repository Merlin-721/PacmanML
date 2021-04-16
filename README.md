# PacmanML
This is my implementation of a Q-Learning agent, using a Q-table, in the Berkeley Pac-Man environment.
## Requirements

* Python 2.7
* Numpy

## Usage
In your command line enter:

```python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid```

This selects the agent QLearnAgent, to train on 2000 games, and run on an additional 10 games (2010-2000=10), on smallGrid layout.

When you've tried this, check out his performance on testClassic.

```python pacman.py -p QLearnAgent -x 2000 -n 2010 -l testClassic```
