from enum import Enum


class BoundaryType(Enum):
    IN = 1
    OUT = 2
    WALL = 3
    FAR = 4
    CYL = 5
    DIRICHLET = 6
    NEUMANN = 7
    SLIP = 8
