"""Action definitions, observation token dimensions, and static array padding limits."""

from __future__ import annotations

# Padding limits for unified JIT compilation across varying field and team sizes (P5)
MAX_CROSSERS: int = 6
MAX_GUARDS: int = 8
MAX_TRANSVERSE_LINES: int = 10
MAX_LONGITUDINAL_LINES: int = 4
MAX_PLAYERS: int = 1 + MAX_CROSSERS + MAX_GUARDS  # 15
MAX_LINES: int = MAX_TRANSVERSE_LINES + MAX_LONGITUDINAL_LINES  # 14

# Feature token dimensions
PLAYER_TOKEN_DIM: int = 27
LINE_TOKEN_DIM: int = 8
GLOBAL_TOKEN_DIM: int = 20

# Discrete Action Head Cardinalities
# Crosser heads: move (9), sprint (2), call (5)
CROSSER_ACTION_DIMS = (9, 2, 5)

# Guard heads: move (3), dash (2), turn (2)
GUARD_ACTION_DIMS = (3, 2, 2)

# Kili heads: move (5), dash (2), turn (2), clap (2)
KILI_ACTION_DIMS = (5, 2, 2, 2)

# Move direction vectors (field coordinate frame: +y = North towards far zone, +x = East)
# 0=stay, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
SQRT2_INV = 0.7071067811865475

CROSSER_MOVE_VECTORS = (
    (0.0, 0.0),  # 0: Stay
    (0.0, 1.0),  # 1: N
    (SQRT2_INV, SQRT2_INV),  # 2: NE
    (1.0, 0.0),  # 3: E
    (SQRT2_INV, -SQRT2_INV),  # 4: SE
    (0.0, -1.0),  # 5: S
    (-SQRT2_INV, -SQRT2_INV),  # 6: SW
    (-1.0, 0.0),  # 7: W
    (-SQRT2_INV, SQRT2_INV),  # 8: NW
)

# Guard move directions: 0=Stay, 1=W (-x), 2=E (+x)
GUARD_MOVE_VECTORS = (
    0.0,  # 0: Stay
    -1.0,  # 1: W
    1.0,  # 2: E
)

# Kili move directions: 0=Stay, 1=N, 2=E, 3=S, 4=W
KILI_MOVE_VECTORS = (
    (0.0, 0.0),  # 0: Stay
    (0.0, 1.0),  # 1: N (+y)
    (1.0, 0.0),  # 2: E (+x)
    (0.0, -1.0),  # 3: S (-y)
    (-1.0, 0.0),  # 4: W (-x)
)
