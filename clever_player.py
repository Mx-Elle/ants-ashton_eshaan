from random import choice

import numpy as np
import numpy.typing as npt

from board import Entity, neighbors, toroidal_distance_2
from random_player import AntMove, valid_neighbors



def toroidal_distance_1(a: tuple[int, int], b: tuple[int, int], shape: tuple[int, int]) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dx = min(dx, shape[0] - dx)
    dy = min(dy, shape[1] - dy)
    return dx + dy

def toroidal_step(
    ant: tuple[int, int],
    target: tuple[int, int],
    walls: npt.NDArray[np.int_],
    forbidden: set[tuple[int, int]] | None = None,
    distance_function = toroidal_distance_2,
) -> tuple[int, int] | None:
    best = None
    best_d = distance_function(ant, target, walls.shape)
    for nb in neighbors(ant, walls.shape):
        if walls[nb]:
            continue
        if forbidden and nb in forbidden:
            continue
        d = distance_function(nb, target, walls.shape)
        if d < best_d:
            best_d = d
            best = nb
    return best


class CleverBot:

    EXPLORE_DIRS: list[tuple[int, int]] = [
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
        self.__name = "CleverBotCareful"

        self.dead_ends: frozenset[tuple[int, int]] = self._compute_dead_ends(1)

    @property
    def name(self) -> str:
        return self.__name
    
    
    def _dist2(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return toroidal_distance_2(a, b, self.walls.shape)
    
    def _dist1(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return toroidal_distance_1(a, b, self.walls.shape)


    def _compute_dead_ends(self, step:int=3) -> frozenset[tuple[int, int]]:
        shape = self.walls.shape
        result: set[tuple[int, int]] = set()

        for x in range(shape[0]):
            for y in range(shape[1]):
                if self.walls[x, y]:
                    continue

                xs = (x+step)%shape[0]
                ys = y 

                count = 0
                smooth = self.walls[xs, ys]
                directions = ((-1,1), (-1,-1), (1,-1), (1,1))
                for i in range(4 *step):
                    direction = directions[i%step] 
                    xs = (xs+direction[0])%shape[0]
                    ys = (ys+direction[1])%shape[1]

                    if self.walls[xs, ys] != smooth:
                        count += 1
                        smooth = not smooth
                
                if count>=2 and count <= 4:
                    result.add((x, y))
                        

        return frozenset(result)

    def _nearest(
        self,
        origin: tuple[int, int],
        targets: set[tuple[int, int]],
        distance_function = toroidal_distance_1,
    ) -> tuple[int, int] | None:
        if not targets:
            return None
        return min(targets, key=lambda t: distance_function(origin, t, self.walls.shape))

    def _hill_is_threatened(
        self,
        hill: tuple[int, int],
        enemy_ants: set[tuple[int, int]],
    ) -> bool:
        lookahead_r1 = (self.battle_radius + 2) 
        return any(self._dist1(e, hill) <= lookahead_r1 for e in enemy_ants)

    def _food_within_reach(
        self,
        ant: tuple[int, int],
        food_cells: set[tuple[int, int]],
    ) -> int:
        r1 = self.collect_radius**2
        return sum(1 for f in food_cells if self._dist2(ant, f) <= r1)

    def _explore_target(
        self,
        ant: tuple[int, int],
        food_cells: set[tuple[int, int]],
        enemy_ants: set[tuple[int, int]],
    ) -> tuple[int, int]:
        
        search_distances = [2 * max(self.vision_radius, self.battle_radius), 
                            3 * max(self.vision_radius, self.battle_radius)]

        gain = -1
        best_target = None
        for search_dist in search_distances:
            for dx, dy in self.EXPLORE_DIRS:
                possible_target = ((ant[0] + dx * search_dist) % self.walls.shape[0],
                        (ant[1] + dy * search_dist) % self.walls.shape[1])
                
                possible_food = self._food_within_reach(possible_target, food_cells)
                possible_threats = sum(1 for h in enemy_ants if self._dist2(possible_target, h) <= self.battle_radius**2)

                score = possible_food - possible_threats
                if score > gain:
                    gain = score
                    best_target = possible_target

        return best_target if best_target is not None else ant 

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
        r2 = self.battle_radius**2
        n_enemies = sum(1 for e in enemy_ants if self._dist2(ant, e) <= r2)

        if n_enemies == 0:
            return self._pick_move(ant, normal_target, claimed_dests, avoid=avoid)

        n_friends = sum(1 for a in my_ants if self._dist2(ant, a) <= r2)

        if n_friends > n_enemies:
            return self._pick_move(ant, normal_target, claimed_dests, avoid=avoid)

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
        
        out = set()
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

        threatened_hills = {
            h for h in my_hills
            if self._hill_is_threatened(h, enemy_ants)
        }

        
        n_ants = len(my_ants) 
        attackers_number = 0.5 + 0.3*(1-1/(n_ants+1)**2) 
        harvesters_number = 0.7 -0.3*(1-1/(n_ants+1)**2) + 0.3*(1-1/(stored_food+1)**2)  
        n_defenders_needed = len(threatened_hills) * 3
        n_harvesters_needed = int(len(dead_end_food) * harvesters_number)  
        spare = n_ants - max(n_defenders_needed, 2) - n_harvesters_needed
        n_attackers = int(spare * attackers_number)  

        ants_sorted = sorted(my_ants)
        assigned: set[tuple[int, int]] = set()

        defender_set: set[tuple[int, int]] = set()
        for hill in threatened_hills:
            candidates = sorted(
                (a for a in ants_sorted if a not in assigned),
                key=lambda a: self._dist2(a, hill),
            )
            for ant in candidates[:3]:
                assigned.add(ant)
                defender_set.add(ant)

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

        claimed_dests: set[tuple[int, int]] = set(my_hills)

        def queue_move(ant: tuple[int, int], dest: tuple[int, int]) -> None:
            if dest != ant:
                out.add((ant, dest))
                claimed_dests.add(dest)


         # Attackers 
        for ant in sorted(attacker_set):
            dest = self._combat_aware_dest(
                ant, attack_target, my_ants, enemy_ants, my_hills, claimed_dests,
                avoid=self.dead_ends,   
            )
            queue_move(ant, dest)


        # Defenders 
        for ant in sorted(defender_set):
            nearest_threat = self._nearest(ant, threatened_hills)
            dest = self._combat_aware_dest(
                ant, nearest_threat, my_ants, enemy_ants, my_hills, claimed_dests
            )
            queue_move(ant, dest)

        # Harvesters 
        for ant, food_target in harvester_targets.items():
            dest = self._pick_move(ant, food_target, claimed_dests)
            queue_move(ant, dest)

       
        # Gatherers 
        for ant in remaining:
            target = self._nearest(ant, open_food)
            if target is None and enemy_hills:
                target = self._nearest(ant, enemy_hills)
            if target is None:
                target = self._explore_target(ant, food, enemy_ants)
            avoid = self.dead_ends if (target not in self.dead_ends) else None
            dest = self._combat_aware_dest(
                ant, target, my_ants, enemy_ants, my_hills, claimed_dests,
                avoid=avoid,
            )
            queue_move(ant, dest)

        return out

    def __str__(self) -> str:
        return self.name