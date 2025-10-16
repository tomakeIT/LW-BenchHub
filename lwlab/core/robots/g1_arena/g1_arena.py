from isaac_arena.embodiments.g1.g1 import G1WBCJointEmbodiment, G1WBCPinkEmbodiment
from lwlab.core.robots.robot_arena_base import LwLabEmbodimentBase


class G1ArenaJointEmbodiment(G1WBCJointEmbodiment, LwLabEmbodimentBase):
    pass


class G1ArenaPinkEmbodiment(G1WBCPinkEmbodiment, LwLabEmbodimentBase):
    pass
