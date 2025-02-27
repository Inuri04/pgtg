from gymnasium.envs.registration import register

from pgtg.environment import PGTGEnv

__version__ = "0.5.0"

register(id="pgtg-v4", entry_point="pgtg.environment:PGTGEnv")
