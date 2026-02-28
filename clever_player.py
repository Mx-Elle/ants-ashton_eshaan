from random import choice
import logging

import numpy as np
import numpy.typing as npt

from board import Entity, neighbors, toroidal_distance_2
from random_player import RandomBot, valid_neighbors

AntMove = tuple[tuple[int, int], tuple[int, int]]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def toroidal_step(
    ant: tuple[int, int],
    target: tuple[int, int],
    walls: npt.NDArray[np.int_],
    forbidden: set[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    shape = walls.shape
    best: tuple[int, int] | None = None
    best_d = toroidal_distance_2(ant, target, shape)

    for nb in neighbors(ant, shape):
        if walls[nb]:
            continue
        if forbidden is not None and nb in forbidden:
            continue
        d = toroidal_distance_2(nb, target, shape)
        if d < best_d:
            best_d = d
            best = nb

    return best



class CleverBot(RandomBot):


    _EXPLORE_DIRS: list[tuple[int, int]] = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0),  (1, -1), (0, -1), (-1, -1),
    ]

    def __init__(
        self,
        walls: npt.NDArray[np.int_],
        harvest_radius: int,
        vision_radius: int,
        battle_radius: int,
        max_turns: int,
        time_per_turn: float,
    ) -> None:
        self.walls = walls
        self.collect_radius = harvest_radius
        self.vision_radius = vision_radius
        self.battle_radius = battle_radius
        self.max_turns = max_turns
        self.time_per_turn = time_per_turn
        self.__name = "CleverBot"

        self.dead_ends: frozenset[tuple[int, int]] = self._compute_dead_ends()
        logger.info(f"Dead-end/corridor cells: {len(self.dead_ends)}")

    @property
    def name(self) -> str:
        return self.__name

    # ------------------------------------------------------------------
    # One-time map analysis
    # ------------------------------------------------------------------

    def _compute_dead_ends(self) -> frozenset[tuple[int, int]]:
        shape = self.walls.shape
        result: set[tuple[int, int]] = set()

        for r in range(shape[0]):
            for c in range(shape[1]):
                if self.walls[r, c]:
                    continue
                passable_nb = [n for n in neighbors((r, c), shape) if not self.walls[n]]
                # FIX: your code had <= 3 which tags almost everything.
                if len(passable_nb) <= 2:
                    result.add((r, c))

        return frozenset(result)


    def _dist2(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return toroidal_distance_2(a, b, self.walls.shape)

    def _nearest(
        self,
        origin: tuple[int, int],
        targets: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if not targets:
            return None
        return min(targets, key=lambda t: self._dist2(origin, t))

    def _hill_is_threatened(
        self,
        hill: tuple[int, int],
        enemy_ants: set[tuple[int, int]],
    ) -> bool:
        lookahead_r2 = (self.battle_radius + 2) ** 2
        return any(self._dist2(e, hill) <= lookahead_r2 for e in enemy_ants)

    def _explore_target(self, ant: tuple[int, int], index: int) -> tuple[int, int]:
        shape = self.walls.shape
        dr, dc = self._EXPLORE_DIRS[index % len(self._EXPLORE_DIRS)]
        reach = max(shape) // 2
        r = (ant[0] + dr * reach) % shape[0]
        c = (ant[1] + dc * reach) % shape[1]
        return (r, c)

    def _pick_move(
        self,
        ant: tuple[int, int],
        target: tuple[int, int] | None,
        claimed_dests: set[tuple[int, int]],
        avoid: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        if target is not None:
            step = toroidal_step(ant, target, self.walls, forbidden=avoid)
            if step is not None and step not in claimed_dests:
                return step

        options = [
            v for v in valid_neighbors(*ant, self.walls)
            if v not in claimed_dests and (avoid is None or v not in avoid)
        ]
        if options:
            return choice(options)

        options_no_avoid = [
            v for v in valid_neighbors(*ant, self.walls)
            if v not in claimed_dests
        ]
        return choice(options_no_avoid) if options_no_avoid else ant

    def _combat_aware_dest(
        self,
        ant: tuple[int, int],
        normal_target: tuple[int, int] | None,
        my_ants: set[tuple[int, int]],
        enemy_ants: set[tuple[int, int]],
        my_hills: set[tuple[int, int]],
        claimed_dests: set[tuple[int, int]],
        avoid: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        r2 = self.battle_radius ** 2
        n_enemies = sum(1 for e in enemy_ants if self._dist2(ant, e) <= r2)

        if n_enemies == 0:
            return self._pick_move(ant, normal_target, claimed_dests, avoid=avoid)

        n_friends = sum(1 for a in my_ants if self._dist2(ant, a) <= r2)

        if n_friends > n_enemies:
            return self._pick_move(ant, normal_target, claimed_dests, avoid=avoid)

        # retreat / regroup
        retreat_to = self._nearest(ant, my_hills)
        if retreat_to is None:
            others = my_ants - {ant}
            retreat_to = self._nearest(ant, others)
        return self._pick_move(ant, retreat_to, claimed_dests)


    def move_ants(
        self,
        vision: set[tuple[tuple[int, int], Entity]],
        stored_food: int,
    ) -> set[AntMove]:

        my_ants     = {c for c, k in vision if k == Entity.FRIENDLY_ANT}
        my_hills    = {c for c, k in vision if k == Entity.FRIENDLY_HILL}
        enemy_ants  = {c for c, k in vision if k == Entity.ENEMY_ANT}
        enemy_hills = {c for c, k in vision if k == Entity.ENEMY_HILL}
        food        = {c for c, k in vision if k == Entity.FOOD}

        hr2 = self.collect_radius ** 2

        contested_food = {
            f for f in food
            if any(self._dist2(e, f) <= hr2 for e in enemy_ants)
        }
        safe_food = food - contested_food

        dead_end_food = safe_food & self.dead_ends
        open_food     = safe_food - dead_end_food

        threatened_hills = {h for h in my_hills if self._hill_is_threatened(h, enemy_ants)}

        n_ants = len(my_ants)
        n_defenders_needed = len(threatened_hills) * 3
        n_harvesters_needed = len(dead_end_food)
        spare = n_ants - max(n_defenders_needed, 2) - n_harvesters_needed
        n_attackers = max(0, spare) // 2

        ants_sorted = sorted(my_ants)
        assigned: set[tuple[int, int]] = set()

        # Defenders
        defender_set: set[tuple[int, int]] = set()
        for hill in threatened_hills:
            candidates = sorted(
                (a for a in ants_sorted if a not in assigned),
                key=lambda a: self._dist2(a, hill),
            )
            for ant in candidates[:3]:
                assigned.add(ant)
                defender_set.add(ant)

        # Harvesters (ant -> food)
        harvester_targets: dict[tuple[int, int], tuple[int, int]] = {}
        for food_cell in sorted(dead_end_food):
            candidates = sorted(
                (a for a in ants_sorted if a not in assigned),
                key=lambda a: self._dist2(a, food_cell),
            )
            if candidates:
                harvester = candidates[0]
                assigned.add(harvester)
                harvester_targets[harvester] = food_cell

        # Attackers
        attacker_set: set[tuple[int, int]] = set()
        attack_target: tuple[int, int] | None = None
        if enemy_hills and n_attackers > 0:
            attack_target = min(
                enemy_hills,
                key=lambda h: min(
                    (self._dist2(a, h) for a in ants_sorted if a not in assigned),
                    default=float("inf"),
                ),
            )
            candidates = sorted(
                (a for a in ants_sorted if a not in assigned),
                key=lambda a: self._dist2(a, attack_target),
            )
            for ant in candidates[:n_attackers]:
                assigned.add(ant)
                attacker_set.add(ant)

        remaining = [a for a in ants_sorted if a not in assigned]

        # Execute moves
        out: set[AntMove] = set()

        claimed_dests: set[tuple[int, int]] = set(my_hills)  # don't block spawners

        def commit(ant: tuple[int, int], dest: tuple[int, int]) -> None:
            claimed_dests.add(dest)
            if dest != ant:
                out.add((ant, dest))

        # Defenders
        for ant in sorted(defender_set):
            nearest_threat = self._nearest(ant, threatened_hills)
            dest = self._combat_aware_dest(
                ant, nearest_threat, my_ants, enemy_ants, my_hills, claimed_dests
            )
            commit(ant, dest)

        # Harvesters
        for ant, food_target in harvester_targets.items():
            dest = self._pick_move(ant, food_target, claimed_dests)
            commit(ant, dest)

        # Attackers (avoid dead-ends)
        for ant in sorted(attacker_set):
            dest = self._combat_aware_dest(
                ant, attack_target, my_ants, enemy_ants, my_hills, claimed_dests,
                avoid=self.dead_ends,
            )
            commit(ant, dest)

        # Gatherers / Explorers
        for idx, ant in enumerate(remaining):
            target = self._nearest(ant, open_food)
            if target is None and enemy_hills:
                target = self._nearest(ant, enemy_hills)
            if target is None:
                target = self._explore_target(ant, idx)

            avoid = self.dead_ends if (target not in self.dead_ends) else None
            dest = self._combat_aware_dest(
                ant, target, my_ants, enemy_ants, my_hills, claimed_dests,
                avoid=avoid,
            )
            commit(ant, dest)

        return out