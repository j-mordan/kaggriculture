import math
from collections import defaultdict
import numpy as np

def step_toward(pos, target):
    """Return a single movement action (NORTH/SOUTH/EAST/WEST) that reduces
    Manhattan distance to target. Locked tiles are passable, so no
    obstacle-avoidance is needed."""
    px, py = pos
    tx, ty = target
    if px == tx and py == ty:
        return None
    dx, dy = tx - px, ty - py
    if abs(dx) > 0:
        return ["EAST"] if dx > 0 else ["WEST"]
    if dy != 0:
        return ["SOUTH"] if dy > 0 else ["NORTH"]
    return ["PASS"]

def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """Calculates the Manhattan distance between two (x, y) coordinates."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def eval_func(func, x, T):
    if func == "linear":
        return x
    elif func == "sq":
        return x * x
    elif func == "sqrt":
        return math.sqrt(x)
    elif func == "log":
        return math.log(1.0 + x)
    elif func == "hinge":
        u = x / T
        return u + 8.0 * (max(0.0, u - 1.0) ** 2)
    raise ValueError(f"Unknown shape function: {func}")

def predict_price(
    expected_inv: int,
    base: float,
    T: float,
    below_func: str,
    below_target: float,
    above_func: str,
    above_target: float,
    I0: int = 10_000
) -> int:
    if expected_inv == I0:
        return max(1, round(base))

    if expected_inv < I0:  # Scarcity
        x = float(I0 - expected_inv)
        f_val = eval_func(below_func, x, T)
        f_T = eval_func(below_func, T, T)
        amp = (below_target * base) / f_T
        raw_price = base + amp * f_val
    else:  # Glut
        x = float(expected_inv - I0)
        f_val = eval_func(above_func, x, T)
        f_T = eval_func(above_func, T, T)
        amp = (above_target * base) / f_T
        raw_price = base - amp * f_val

    return max(1, round(raw_price))

def build_chunks(products, seed_cost, prices, projected_harvest, projected_inventory,
                  base_costs, T_values, below_funcs, below_targets, above_funcs, above_targets,
                  product_lifespan, product_cost_per_day, budget, max_units):
    """One chunk per potential purchase: (product, cost, marginal_daily_profit).
    Values are strictly non-increasing per product because buying more pushes
    its price down via predict_price, mirroring what update_value already does."""
    chunks = []
    for p in products:
        inv = projected_inventory[p]
        extra = prices["WHEAT"] if p in ("EGG", "MILK", "WOOL") else 0
        cost = seed_cost[p] + extra
        if cost > budget or product_lifespan[p] <= 0:
            continue
        for _unit in range(max_units):
            inv += projected_harvest[p]
            price = predict_price(inv, base_costs[p], T_values[p], below_funcs[p],
                                   below_targets[p], above_funcs[p], above_targets[p])
            rev = price * projected_harvest[p] / product_lifespan[p]
            value = rev - product_cost_per_day[p]
            if value <= 0:
                break
            chunks.append((p, cost, value))
    return chunks

def solve_allocation_np(chunks, budget, tiles):
    """
    chunks: list of (product, cost, value) tuples, one per potential unit purchase,
            with value non-increasing per product (from build_chunks()).
    budget, tiles: integers.
    Returns (bought: dict[product]->count, total_daily_profit: float)
    """
    budget = int(budget)
    if budget < 0:
        budget = 0

    NEG = -1e18
    dp = np.full((tiles + 1, budget + 1), NEG, dtype=np.float64)
    dp[0, 0] = 0.0
    # One (tiles x budget) array total -- NOT one per chunk, so memory stays small
    picked = np.full((tiles + 1, budget + 1), -1, dtype=np.int32)

    for idx, (p, cost, value) in enumerate(chunks):
        cost = int(round(cost))
        if cost > budget or cost <= 0:
            continue
        for t in range(tiles, 0, -1):
            prev_row = dp[t - 1, :budget - cost + 1]     # states reachable before this chunk
            cur_slice = dp[t, cost:budget + 1]            # states after adding this chunk
            candidate = prev_row + value
            mask = candidate > cur_slice
            dp[t, cost:budget + 1] = np.where(mask, candidate, cur_slice)
            picked[t, cost:budget + 1] = np.where(mask, idx, picked[t, cost:budget + 1])

    best_t, best_b = np.unravel_index(np.argmax(dp), dp.shape)
    best_v = dp[best_t, best_b]

    bought = defaultdict(int)
    t, b = int(best_t), int(best_b)
    while picked[t, b] != -1:
        idx = int(picked[t, b])
        p, cost, value = chunks[idx]
        bought[p] += 1
        t, b = t - 1, b - int(round(cost))
    return bought, float(best_v)

def agent(obs):
    """
    Main entry point called by the Kaggle simulation environment each turn.
    Returns the action commands for units and the market.
    """
    player_id = obs["player"]
    day = obs["day"]
    hour = obs["hour"]
    my_farm = obs["farms"][player_id]
    opp_farm = obs["farms"][1 - player_id]
    shops = obs["town"]["unlocked_shops"]
    my_tiles = my_farm["tiles"]
    private = obs["private"]
    inventories = private["inventories"]
    seeds = private["seeds"]
    shed = private["shed"]
    prices = obs["market"]["prices"]

    day_next_market = 3 * (day // 3 + 1)

    cost_of_n_farmers = {
        1 : 0,
        2 : 1,
        3: 2,
        4: 4,
        5: 7,
        6: 12,
        7: 20,
        8: 33,
        9: 54,
        10: 88,
        11: 143,
        12: 232,
        13: 376,
        14: 609,
        15: 986,
        16: 1596,
        17: 2583,
        18: 4180,
        19: 6767,
        20: 10945
    }

    shed_adjacent_tiles = [(4,4), (5,4), (4,5), (5,5)]

    daily_demand = { #default
        "WHEAT" : 1,
        "CARROT" : 1,
        "MELON" : 1,
        "TOMATO" : 1,
        "STRAWBERRY" : 1,
        "EGG" : 1,
        "MILK" : 1,
        "WOOL" : 1,
        "FERTILIZER" : 0
    }
    for shop in shops:
        if shop == "BAKERY":
            daily_demand["WHEAT"] += 6
            daily_demand["EGG"] += 6
        elif shop =="PIZZA_SHOP":
            daily_demand["WHEAT"] += 6
            daily_demand["TOMATO"] += 6
            daily_demand["MILK"] += 6
        elif shop == "BRUNCH_SPOT":
            daily_demand["EGG"] += 6
            daily_demand["WHEAT"] += 6
            daily_demand["STRAWBERRY"] += 6
        elif shop == "YARN_STORE":
            daily_demand["WOOL"] += 12
        elif shop == "ICE_CREAM_SHOP":
            daily_demand["MILK"] += 6
            daily_demand["STRAWBERRY"] += 6
            daily_demand["WHEAT"] += 6
        elif shop == "PET_CAFE":
            daily_demand["CARROT"] += 12
        elif shop == "SMOOTHIE_SHOP":
            daily_demand["MILK"] += 6
            daily_demand["STRAWBERRY"] += 6
        elif shop == "FARMERS_MARKET":
            daily_demand["WHEAT"] += 6
            daily_demand["CARROT"] += 6
            daily_demand["TOMATO"] += 6
            daily_demand["STRAWBERRY"] += 6
        else:
            print('something went wrong with shop:', shop)

    demand_EV_Day = { #every time a shop opens, expected demand spikes by this much per day
        "WHEAT" : 3.75,
        "CARROT" : 2.25,
        "MELON" : 0,
        "TOMATO" : 1.5,
        "STRAWBERRY" : 3,
        "EGG" : 1.5,
        "MILK" : 2.25,
        "WOOL" : 1.5,
        "FERTILIZER" : 0
    }

    projected_produced = {
        "WHEAT" : 0,
        "CARROT" : 0,
        "MELON" : 0,
        "TOMATO" : 0,
        "STRAWBERRY" : 0,
        "EGG" : 0,
        "MILK" : 0,
        "WOOL" : 0,
        "FERTILIZER" : 0
    }

    max_harvest = {
        "WHEAT" : 6,
        "CARROT" : 4,
        "MELON" : 6,
        "TOMATO" : 4, #max held?
        "STRAWBERRY" : 4,
        "EGG" : 4,
        "MILK" : 6, 
        "WOOL" : 6,
        "FERTILIZER" : 30 - day
    }

    projected_harvest = {
        "WHEAT" : 4,
        "CARROT" : 3,
        "MELON" : 6,
        "TOMATO" : 7,
        "STRAWBERRY" : 8,
        "EGG" : 4 + 2 * (25 - day) if day < 25 else 0,
        "MILK" : 6 + 3 * (21 - day) // 2 if day < 21 else 0, 
        "WOOL" : 6 + 4 * (23 - day) // 3 if day < 23 else 0,
        "FERTILIZER" : 30 - day
    }

    product_lifespan = {
            "WHEAT" : 4,
            "CARROT" : 3,
            "MELON" : 10,
            "TOMATO" : 11,
            "STRAWBERRY" : 16,
            "EGG" : 30 - day,
            "MILK" : 30 - day,
            "WOOL" : 30 - day,
            "FERTILIZER" : 30 - day
    }

    
    time_to_first_yield = {
        "WHEAT" : 2,
        "CARROT" : 2,
        "MELON" : 10,
        "TOMATO" : 8,
        "STRAWBERRY" : 10,
        "EGG" : 4,
        "GOOSE" : 4,
        "MILK" : 8,
        "COW" : 8,
        "WOOL" : 6,
        "SHEEP" : 6,
        "FERTILIZER" : 1
    }

    seed_cost = {
        "WHEAT" : 10,
        "CARROT" : 20,
        "MELON" : 80,
        "TOMATO" : 50,
        "STRAWBERRY" : 100,
        "EGG" : 300 + 2 * prices["WHEAT"],
        "MILK" : 400 + 2 * prices["WHEAT"],
        "WOOL" : 500 + 2 * prices["WHEAT"]
    }

    new_quadrant_costs = {
        2 : 1000,
        3 : 2000,
        4: 4000
    }

    empty_tiles_in_col = {
        (0, 0): 0,
        (1, 0): 0,
        (2, 0): 0,
        (3, 0): 0,
        (4, 0): 0,
        (5, 0): 0,
        (6, 0): 0,
        (7, 0): 0,
        (8, 0): 0,
        (9, 0): 0,
        (0, 9): 0,
        (1, 9): 0,
        (2, 9): 0,
        (3, 9): 0,
        (4, 9): 0,
        (5, 9): 0,
        (6, 9): 0,
        (7, 9): 0,
        (8, 9): 0,
        (9, 9): 0,        
    }
    empty_pastures_in_col = {
        (0, 0): 0,
        (1, 0): 0,
        (2, 0): 0,
        (3, 0): 0,
        (4, 0): 0,
        (5, 0): 0,
        (6, 0): 0,
        (7, 0): 0,
        (8, 0): 0,
        (9, 0): 0,
        (0, 9): 0,
        (1, 9): 0,
        (2, 9): 0,
        (3, 9): 0,
        (4, 9): 0,
        (5, 9): 0,
        (6, 9): 0,
        (7, 9): 0,
        (8, 9): 0,
        (9, 9): 0,   
    }
    empty_coops_in_col = {
        (0, 0): 0,
        (1, 0): 0,
        (2, 0): 0,
        (3, 0): 0,
        (4, 0): 0,
        (5, 0): 0,
        (6, 0): 0,
        (7, 0): 0,
        (8, 0): 0,
        (9, 0): 0,
        (0, 9): 0,
        (1, 9): 0,
        (2, 9): 0,
        (3, 9): 0,
        (4, 9): 0,
        (5, 9): 0,
        (6, 9): 0,
        (7, 9): 0,
        (8, 9): 0,
        (9, 9): 0,   
        }
    animals_in_col = {
        (0, 0): 0,
        (1, 0): 0,
        (2, 0): 0,
        (3, 0): 0,
        (4, 0): 0,
        (5, 0): 0,
        (6, 0): 0,
        (7, 0): 0,
        (8, 0): 0,
        (9, 0): 0,
        (0, 9): 0,
        (1, 9): 0,
        (2, 9): 0,
        (3, 9): 0,
        (4, 9): 0,
        (5, 9): 0,
        (6, 9): 0,
        (7, 9): 0,
        (8, 9): 0,
        (9, 9): 0,   
    }

    fert_needed_in_col = {
        (0, 0): [],
        (1, 0): [],
        (2, 0): [],
        (3, 0): [],
        (4, 0): [],
        (5, 0): [],
        (6, 0): [],
        (7, 0): [],
        (8, 0): [],
        (9, 0): [],
        (0, 9): [],
        (1, 9): [],
        (2, 9): [],
        (3, 9): [],
        (4, 9): [],
        (5, 9): [],
        (6, 9): [],
        (7, 9): [],
        (8, 9): [],
        (9, 9): []
    }

    poop_in_col = {
        (0, 0): [],
        (1, 0): [],
        (2, 0): [],
        (3, 0): [],
        (4, 0): [],
        (5, 0): [],
        (6, 0): [],
        (7, 0): [],
        (8, 0): [],
        (9, 0): [],
        (0, 9): [],
        (1, 9): [],
        (2, 9): [],
        (3, 9): [],
        (4, 9): [],
        (5, 9): [],
        (6, 9): [],
        (7, 9): [],
        (8, 9): [],
        (9, 9): []
    }
        

    farmer_col = { #default, incorrect because market orders happen in multiple turns
        0: (0, 0),
        1: (1, 0),
        2: (3, 0),
        3: (4, 0),
        4: (2, 0),
        5: (9, 0),
        6: (5, 0),
        7: (7, 0),
        8: (6, 0),
        9: (8, 0),
        10: (3, 9),
        11: (0, 9),
        12: (1, 9),
        13: (2, 9),
        14: (4, 9),
        15: (8, 9),
        16: (9, 9),
        17: (5, 9),
        18: (7, 9),
        19: (6, 9)
    }

    '''
    columns_needed = []
    unlocked_quads = len(my_farm["unlocked_quadrants"])
    for i in range(unlocked_quads * 5):
        columns_needed.append(farmer_col[i])
    for c in columns_needed:
        if len(my_farm["hands"]) + 1 < unlocked_quads * 5:
            if unlocked_quads == 1:
                min_dist = 100
                min_ind = 0
                for j in len(my_farm["hands"]) + 1:
                    current_pos = tuple(farmer_pos if j == 0 else my_farm["hands"][j - 1])
                    if manhattan_distance(current_pos, c) < min_dist:
                        min_dist = manhattan_distance(current_pos, c)
                        min_ind = j
    '''



                        


    empty_tile_list = []
    empty_pasture_list = []
    empty_coop_list = []

    total_animals_both_farms = 0
    animals_my_farm = 0
    harvestable = []
    empty_tiles = 0
    for y, tile_row in enumerate(my_tiles):
        for x, tile in enumerate(tile_row):
            if tile == None:
                empty_tiles += 1
                if y <= 4:
                    empty_tiles_in_col[(x, 0)] += 1
                else:
                    empty_tiles_in_col[(x, 9)] += 1
                empty_tile_list.append((x, y))
                continue
            elif tile == "LOCKED":
                continue
            elif tile.get("kind") == "PLANT":
                age = day - tile.get("planted_day", 0)
                max_harvest_size = max_harvest[tile.get("crop")]
                first_yield_day = time_to_first_yield[tile.get("crop")]
                lifespan = product_lifespan[tile.get("crop")]
                bonus_day_start = math.ceil(lifespan / 2)
                crop_price = prices[tile.get("crop")]
                if tile.get("yield_units") > 0 and age >= first_yield_day:
                    harvestable.append((x, y))
                if tile.get("crop") == "WHEAT":
                    projected_produced["WHEAT"] += projected_harvest["WHEAT"]
                    if age == bonus_day_start and tile.get("fertilized_until_day") == -1 and prices["FERTILIZER"] < 2 * crop_price:
                        if y <= 4:
                            fert_needed_in_col[(x, 0)].append((x, y))
                        else:
                            fert_needed_in_col[(x, 9)].append((x, y))
                    if age >= lifespan or (tile.get("yield_units") >= max_harvest_size and age >= first_yield_day):
                        empty_tiles += 1
                        if y <= 4:
                            empty_tiles_in_col[(x, 0)] += 1
                        else:
                            empty_tiles_in_col[(x, 9)] += 1
                        empty_tile_list.append((x, y))
                    #if tile.get("fertilized_until_day") == -1:
                        #projected_produced["FERTILIZER"] -= 1
                elif tile.get("crop") == "CARROT":
                    projected_produced["CARROT"] += projected_harvest["CARROT"]
                    if age == bonus_day_start and tile.get("fertilized_until_day") == -1 and prices["FERTILIZER"] < crop_price:
                        if y <= 4:
                            fert_needed_in_col[(x, 0)].append((x, y))
                        else:
                            fert_needed_in_col[(x, 9)].append((x, y))
                    if age >= lifespan or (tile.get("yield_units") >= max_harvest_size and age >= first_yield_day):
                        empty_tiles += 1
                        if y <= 4:
                            empty_tiles_in_col[(x, 0)] += 1
                        else:
                            empty_tiles_in_col[(x, 9)] += 1
                        empty_tile_list.append((x, y))
                    #if tile.get("fertilized_until_day") == -1:
                        #projected_produced["FERTILIZER"] -= 1
                elif tile.get("crop") == "MELON":
                    projected_produced["MELON"] += projected_harvest["MELON"]
                    if age >= lifespan or (tile.get("yield_units") >= max_harvest_size and age >= first_yield_day):
                        empty_tiles += 1
                        if y <= 4:
                            empty_tiles_in_col[(x, 0)] += 1
                        else:
                            empty_tiles_in_col[(x, 9)] += 1
                        empty_tile_list.append((x, y))
                elif tile.get("crop") == "TOMATO":
                    if age <= 8:
                        projected_produced["TOMATO"] += projected_harvest["TOMATO"]
                    elif age <= 9:
                        projected_produced["TOMATO"] += projected_harvest["TOMATO"] - 2
                    elif age <= 10:
                        projected_produced["TOMATO"] += projected_harvest["TOMATO"] - 4
                    elif age <= 11:
                        projected_produced["TOMATO"] += projected_harvest["TOMATO"] - 6
                    if (age == 7 or age == 8) and tile.get("fertilized_until_day") == -1 and prices["FERTILIZER"] < 3 * crop_price:
                        if y <= 4:
                            fert_needed_in_col[(x, 0)].append((x, y))
                        else:
                            fert_needed_in_col[(x, 9)].append((x, y))
                    if age >= lifespan:
                        empty_tiles += 1
                        if y <= 4:
                            empty_tiles_in_col[(x, 0)] += 1
                        else:
                            empty_tiles_in_col[(x, 9)] += 1
                        empty_tile_list.append((x, y))
                    is_fertilized_now = tile.get("fertilized_until_day", -1) >= day
                    if age < 8:
                        projected_produced["FERTILIZER"] -= 1
                    elif age == 8 and not is_fertilized_now:
                        projected_produced["FERTILIZER"] -= 1
                elif tile.get("crop") == "STRAWBERRY":
                    if age <= 10:
                        projected_produced["STRAWBERRY"] += projected_harvest["STRAWBERRY"]
                    elif age <= 12:
                        projected_produced["STRAWBERRY"] += projected_harvest["STRAWBERRY"] - 2
                    elif age <= 14:
                        projected_produced["STRAWBERRY"] += projected_harvest["STRAWBERRY"] - 4
                    elif age <= 16:
                        projected_produced["STRAWBERRY"] += projected_harvest["STRAWBERRY"] - 6
                    if age >= 9 and (tile.get("fertilized_until_day") == -1  or tile.get("fertilized_until_day") < day) and prices["FERTILIZER"] < 2 * crop_price:
                        if y <= 4:
                            fert_needed_in_col[(x, 0)].append((x, y))
                        else:
                            fert_needed_in_col[(x, 9)].append((x, y))
                    if age >= lifespan:
                        empty_tiles += 1
                        if y <= 4:
                            empty_tiles_in_col[(x, 0)] += 1
                        else:
                            empty_tiles_in_col[(x, 9)] += 1
                        empty_tile_list.append((x, y))
                    is_fertilized_now = tile.get("fertilized_until_day", -1) >= day
                    if age < 10:
                        projected_produced["FERTILIZER"] -= 2
                    elif age < 13:
                        if not is_fertilized_now:
                            projected_produced["FERTILIZER"] -= 2
                        else:
                            projected_produced["FERTILIZER"] -= 1
                    elif age == 13:
                        projected_produced["FERTILIZER"] -= 1
                    elif age < 17:
                        if not is_fertilized_now:
                            projected_produced["FERTILIZER"] -= 1
                    
            elif tile.get("kind") == "WEED":
                empty_tiles += 1
                if y <= 4:
                    empty_tiles_in_col[(x, 0)] += 1
                else:
                    empty_tiles_in_col[(x, 9)] += 1
                empty_tile_list.append((x, y))
            elif tile.get("kind") == "COOP":
                if tile.get("yield_units", 0) > 0:
                    harvestable.append((x, y))
                if tile.get("animal") == "GOOSE":
                    projected_produced["EGG"] += projected_harvest["EGG"] - ((day - tile.get("placed_day")) if day - tile.get("placed_day") < 4 else 0)
                    projected_produced["FERTILIZER"] += projected_harvest["FERTILIZER"]
                    total_animals_both_farms += 1
                    animals_my_farm += 1
                    if y <= 4:
                        animals_in_col[(x, 0)] += 1
                        poop_in_col[(x, 0)].append((x, y))
                    else:
                        animals_in_col[(x, 9)] += 1
                        poop_in_col[(x, 9)].append((x, y))
                elif tile.get("animal") == None:
                    if y <= 4:
                        empty_coops_in_col[(x, 0)] += 1
                    else:
                        empty_coops_in_col[(x, 9)] += 1
                    empty_coop_list.append((x, y))
            elif tile.get("kind") == "PASTURE":
                if tile.get("yield_units", 0) > 0:
                    harvestable.append((x, y))
                if tile.get("animal") == "COW":
                    projected_produced["MILK"] += projected_harvest["MILK"] - ((day - tile.get("placed_day")) // 2 if day - tile.get("placed_day") < 8 else 0)
                    projected_produced["FERTILIZER"] += projected_harvest["FERTILIZER"]
                    total_animals_both_farms += 1
                    animals_my_farm += 1
                    if y <= 4:
                        animals_in_col[(x, 0)] += 1
                        poop_in_col[(x, 0)].append((x, y))
                    else:
                        animals_in_col[(x, 9)] += 1
                        poop_in_col[(x, 9)].append((x, y))
                elif tile.get("animal") == "SHEEP":
                    projected_produced["WOOL"] += projected_harvest["WOOL"] - ((day - tile.get("placed_day")) // 3 if day - tile.get("placed_day") < 6 else 0)
                    projected_produced["FERTILIZER"] += projected_harvest["FERTILIZER"]
                    total_animals_both_farms += 1
                    animals_my_farm += 1
                    if y <= 4:
                        animals_in_col[(x, 0)] += 1
                        poop_in_col[(x, 0)].append((x, y))
                    else:
                        animals_in_col[(x, 9)] += 1
                        poop_in_col[(x, 9)].append((x, y))
                elif tile.get("animal") == None:
                    if y <= 4:
                        empty_pastures_in_col[(x, 0)] += 1
                    else:
                        empty_pastures_in_col[(x, 9)] += 1
                    empty_pasture_list.append((x, y))
            else:
                print('unexpected tile kind:', tile.get("kind"))
                continue

    for tile_row in opp_farm["tiles"]:
        for tile in tile_row:
            if tile == None:
                continue
            elif tile == "LOCKED":
                continue
            elif tile.get("kind") == "PLANT":
                if tile.get("crop") == "WHEAT":
                    projected_produced["WHEAT"] += projected_harvest["WHEAT"]
                elif tile.get("crop") == "CARROT":
                    projected_produced["CARROT"] += projected_harvest["CARROT"]
                elif tile.get("crop") == "MELON":
                    projected_produced["MELON"] += projected_harvest["MELON"]
                elif tile.get("crop") == "TOMATO":
                    projected_produced["TOMATO"] += projected_harvest["TOMATO"]
                elif tile.get("crop") == "STRAWBERRY":
                    projected_produced["STRAWBERRY"] += projected_harvest["STRAWBERRY"]
            elif tile.get("kind") == "WEED":
                continue
            elif tile.get("kind") == "COOP":
                if tile.get("animal") == "GOOSE":
                    projected_produced["EGG"] += projected_harvest["EGG"] - ((day - tile.get("placed_day")) if day - tile.get("placed_day") < 4 else 0)
                    projected_produced["FERTILIZER"] += projected_harvest["FERTILIZER"]
                    total_animals_both_farms += 1
            elif tile.get("kind") == "PASTURE":
                if tile.get("animal") == "COW":
                    projected_produced["MILK"] += projected_harvest["MILK"] - ((day - tile.get("placed_day")) // 2 if day - tile.get("placed_day") < 8 else 0)
                    projected_produced["FERTILIZER"] += projected_harvest["FERTILIZER"]
                    total_animals_both_farms += 1
                elif tile.get("animal") == "SHEEP":
                    projected_produced["WOOL"] += projected_harvest["WOOL"] - ((day - tile.get("placed_day")) // 3 if day - tile.get("placed_day") < 6 else 0)
                    projected_produced["FERTILIZER"] += projected_harvest["FERTILIZER"]
                    total_animals_both_farms += 1
            else:
                print('unexpected opp farm tile kind:', tile.get("kind"))
                continue
    for key, cords in poop_in_col.items():
        for (x, y) in sorted(cords, key=lambda pt: pt[1]):
            for (x1, y1) in sorted(fert_needed_in_col[key], key=lambda pt: pt[1]):
                bleh, y_end = key
                if y_end == 0:
                    if y1 > y:
                        fert_needed_in_col[key].remove((x1, y1))
                        break
                else:
                    if y1 < y:
                        fert_needed_in_col[key].remove((x1, y1))
                        break


    current_demand_minus_projected_supply = {
        "WHEAT" : daily_demand["WHEAT"] * product_lifespan["WHEAT"] - projected_produced["WHEAT"],
        "CARROT" : daily_demand["CARROT"] * product_lifespan["CARROT"] - projected_produced["CARROT"],
        "MELON" : daily_demand["MELON"] * product_lifespan["MELON"] - projected_produced["MELON"],
        "TOMATO" : daily_demand["TOMATO"] * product_lifespan["TOMATO"] - projected_produced["TOMATO"],
        "STRAWBERRY" : daily_demand["STRAWBERRY"] * product_lifespan["STRAWBERRY"] - projected_produced["STRAWBERRY"],
        "EGG" : daily_demand["EGG"] * product_lifespan["EGG"] - projected_produced["EGG"],
        "MILK" : daily_demand["MILK"] * product_lifespan["MILK"] - projected_produced["MILK"],
        "WOOL" : daily_demand["WOOL"] * product_lifespan["WOOL"] - projected_produced["WOOL"],
        "FERTILIZER" : daily_demand["FERTILIZER"] - projected_produced["FERTILIZER"]
    }

    days_until_next_market = day_next_market - day
    remaining_egg_days = range(4, min(31, 30 - day + 1))
    remaining_milk_days = range(8, min(31, 30 - day + 1), 2)
    remaining_wool_days = range(6, min(31, 30 - day + 1), 3)

    projected_demand_increase = {
        "WHEAT" : demand_EV_Day["WHEAT"] * sum(range(max(0, 4 - days_until_next_market), 0, -3)),
        "CARROT" : demand_EV_Day["CARROT"] * sum(range(max(0, 3 - days_until_next_market), 0, -3)),
        "MELON" : demand_EV_Day["MELON"] * sum(range(max(0, 10 - days_until_next_market), 0, -3)),
        "TOMATO" : demand_EV_Day["TOMATO"] * (
    sum(range(max(0, 8 - days_until_next_market), 0, -3)) +
    sum(range(max(0, 9 - days_until_next_market), 0, -3)) +
    sum(range(max(0, 10 - days_until_next_market), 0, -3)) +
    sum(range(max(0, 11 - days_until_next_market), 0, -3))
    ) / 4, #this is just an average of the demand increase, but if necessary, this should be upgraded to a real measure of how demand shift changes prices for each individual day
        "STRAWBERRY" : demand_EV_Day["STRAWBERRY"] * (
    sum(range(max(0, 10 - days_until_next_market), 0, -3)) +
    sum(range(max(0, 12 - days_until_next_market), 0, -3)) +
    sum(range(max(0, 14 - days_until_next_market), 0, -3)) +
    sum(range(max(0, 16 - days_until_next_market), 0, -3))
    ) / 4,
        "EGG" : demand_EV_Day["EGG"] * sum(
    sum(range(max(0, h - days_until_next_market), 0, -3)) for h in remaining_egg_days
    ) / len(remaining_egg_days) if remaining_egg_days else 0,
        "MILK" : demand_EV_Day["MILK"] * sum(
    sum(range(max(0, h - days_until_next_market), 0, -3)) for h in remaining_milk_days
    ) / len(remaining_milk_days) if remaining_milk_days else 0,
        "WOOL" : demand_EV_Day["WOOL"] * sum(
    sum(range(max(0, h - days_until_next_market), 0, -3)) for h in remaining_wool_days
    ) / len(remaining_wool_days) if remaining_wool_days else 0,
        "FERTILIZER" : 0
    }

    total_demand_minus_supply = {
        "WHEAT" : current_demand_minus_projected_supply["WHEAT"] + projected_demand_increase["WHEAT"] + 4 * total_animals_both_farms,
        "CARROT" : current_demand_minus_projected_supply["CARROT"] + projected_demand_increase["CARROT"],
        "MELON" : current_demand_minus_projected_supply["MELON"] + projected_demand_increase["MELON"],
        "TOMATO" : current_demand_minus_projected_supply["TOMATO"] + projected_demand_increase["TOMATO"],
        "STRAWBERRY" : current_demand_minus_projected_supply["STRAWBERRY"] + projected_demand_increase["STRAWBERRY"],
        "EGG" : current_demand_minus_projected_supply["EGG"] + projected_demand_increase["EGG"],
        "MILK" : current_demand_minus_projected_supply["MILK"] + projected_demand_increase["MILK"],
        "WOOL" : current_demand_minus_projected_supply["WOOL"] + projected_demand_increase["WOOL"],
        "FERTILIZER" : current_demand_minus_projected_supply["FERTILIZER"] + projected_demand_increase["FERTILIZER"]
    }

    projected_inventory = {
        "WHEAT" : obs["market"]["inventory"].get("WHEAT", 0) - total_demand_minus_supply["WHEAT"],
        "CARROT" : obs["market"]["inventory"].get("CARROT", 0) - total_demand_minus_supply["CARROT"],
        "MELON" : obs["market"]["inventory"].get("MELON", 0) - total_demand_minus_supply["MELON"],
        "TOMATO" : obs["market"]["inventory"].get("TOMATO", 0) - total_demand_minus_supply["TOMATO"],
        "STRAWBERRY" : obs["market"]["inventory"].get("STRAWBERRY", 0) - total_demand_minus_supply["STRAWBERRY"],
        "EGG" : obs["market"]["inventory"].get("EGG", 0) - total_demand_minus_supply["EGG"],
        "MILK" : obs["market"]["inventory"].get("MILK", 0) - total_demand_minus_supply["MILK"],
        "WOOL" : obs["market"]["inventory"].get("WOOL", 0) - total_demand_minus_supply["WOOL"],
        "FERTILIZER" : obs["market"]["inventory"].get("FERTILIZER", 0) - total_demand_minus_supply["FERTILIZER"]
    }

    base_costs = {
        "WHEAT" : 25,
        "CARROT" : 35,
        "MELON" : 250,
        "TOMATO" : 60,
        "STRAWBERRY" : 120,
        "EGG" : 50,
        "MILK" : 160,
        "WOOL" : 200,
        "FERTILIZER" : 100
    }

    T_values = {
        "WHEAT" : 400,
        "CARROT" : 450,
        "MELON" : 300,
        "TOMATO" : 200,
        "STRAWBERRY" : 100,
        "EGG" : 332,
        "MILK" : 122,
        "WOOL" : 105,
        "FERTILIZER" : 200
    }

    below_funcs = {
        "WHEAT" : "sqrt",
        "CARROT" : "hinge",
        "MELON" : "log",
        "TOMATO" : "hinge",
        "STRAWBERRY" : "sqrt",
        "EGG" : "hinge",
        "MILK" : "sqrt",
        "WOOL" : "log",
        "FERTILIZER" : "linear"
    }

    below_targets = {
        "WHEAT" : 0.80,
        "CARROT" : 1.00,
        "MELON" : 0.20,
        "TOMATO" : 0.40,
        "STRAWBERRY" : 0.70,
        "EGG" : 0.40,
        "MILK" : 0.60,
        "WOOL" : 0.20,
        "FERTILIZER" : 0.40
    }

    above_funcs = {
        "WHEAT" : "log",
        "CARROT" : "sqrt",
        "MELON" : "sq",
        "TOMATO" : "sqrt",
        "STRAWBERRY" : "linear",
        "EGG" : "log",
        "MILK" : "linear",
        "WOOL" : "sq",
        "FERTILIZER" : "linear"
    }

    above_targets = {
        "WHEAT" : 0.20,
        "CARROT" : 0.70,
        "MELON" : 3.6,
        "TOMATO" : 0.60,
        "STRAWBERRY" : 1.60,
        "EGG" : 0.20,
        "MILK" : 1.60,
        "WOOL" : 3.20,
        "FERTILIZER" : 0.40
    }

    projected_prices = {
        "WHEAT" : predict_price(projected_inventory["WHEAT"], base_costs["WHEAT"], T_values["WHEAT"], below_funcs["WHEAT"], below_targets["WHEAT"], above_funcs["WHEAT"], above_targets["WHEAT"]) if day < 30 - product_lifespan["WHEAT"] else 0,
        "CARROT" : predict_price(projected_inventory["CARROT"], base_costs["CARROT"], T_values["CARROT"], below_funcs["CARROT"], below_targets["CARROT"], above_funcs["CARROT"], above_targets["CARROT"]) if day < 30 - product_lifespan["CARROT"] else 0,
        "MELON" : predict_price(projected_inventory["MELON"], base_costs["MELON"], T_values["MELON"], below_funcs["MELON"], below_targets["MELON"], above_funcs["MELON"], above_targets["MELON"]) if day < 30 - product_lifespan["MELON"] else 0,
        "TOMATO" : predict_price(projected_inventory["TOMATO"], base_costs["TOMATO"], T_values["TOMATO"], below_funcs["TOMATO"], below_targets["TOMATO"], above_funcs["TOMATO"], above_targets["TOMATO"]) if day < 30 - product_lifespan["TOMATO"] else 0,
        "STRAWBERRY" : predict_price(projected_inventory["STRAWBERRY"], base_costs["STRAWBERRY"], T_values["STRAWBERRY"], below_funcs["STRAWBERRY"], below_targets["STRAWBERRY"], above_funcs["STRAWBERRY"], above_targets["STRAWBERRY"]) if day < 30 - product_lifespan["STRAWBERRY"] else 0,
        "EGG" : predict_price(projected_inventory["EGG"], base_costs["EGG"], T_values["EGG"], below_funcs["EGG"], below_targets["EGG"], above_funcs["EGG"], above_targets["EGG"]),
        "MILK" : predict_price(projected_inventory["MILK"], base_costs["MILK"], T_values["MILK"], below_funcs["MILK"], below_targets["MILK"], above_funcs["MILK"], above_targets["MILK"]),
        "WOOL" : predict_price(projected_inventory["WOOL"], base_costs["WOOL"], T_values["WOOL"], below_funcs["WOOL"], below_targets["WOOL"], above_funcs["WOOL"], above_targets["WOOL"]),
        "FERTILIZER" : predict_price(projected_inventory["FERTILIZER"], base_costs["FERTILIZER"], T_values["FERTILIZER"], below_funcs["FERTILIZER"], below_targets["FERTILIZER"], above_funcs["FERTILIZER"], above_targets["FERTILIZER"])
    }

    '''product_values = { #subtract value based on how many of that crop we have, later
            "WHEAT" : ((daily_demand["WHEAT"] + demand_EV_Day["WHEAT"] * (30 - day)) * prices["WHEAT"]) * 6 / 4,
            "CARROT" : ((daily_demand["CARROT"] + demand_EV_Day["CARROT"] * (30 - day)) * prices["CARROT"]) * 4 / 3,
            "MELON" : ((daily_demand["MELON"]  + demand_EV_Day["MELON"] * (30 - day))  * prices["MELON"]) * 6 / 10,
            "TOMATO" : ((daily_demand["TOMATO"] + demand_EV_Day["TOMATO"] * (30 - day)) * prices["TOMATO"]) * 8 / 11,
            "STRAWBERRY" : ((daily_demand["STRAWBERRY"] + demand_EV_Day["STRAWBERRY"] * (30 - day))  * prices["STRAWBERRY"]) * 8 / 16,
            "EGG" : ((daily_demand["EGG"] + demand_EV_Day["EGG"] * (30 - day)) * prices["EGG"]) * (26 - day) / ((30 - day) if 30 - day > 0 else 1),
            "MILK" : ((daily_demand["MILK"] + demand_EV_Day["MILK"] * (30 - day)) * prices["MILK"]) * (22 - day) / ((30 - day) if 30 - day > 0 else 1),
            "WOOL" : ((daily_demand["WOOL"] + demand_EV_Day["WOOL"] * (30 - day)) * prices["WOOL"]) * (24 - day) / ((30 - day) if 30 - day > 0 else 1)
        }'''

    product_revenue_per_day = {
        "WHEAT" : projected_prices["WHEAT"] * projected_harvest["WHEAT"] / product_lifespan["WHEAT"] if 30 - day > time_to_first_yield["WHEAT"] else 0,
        "CARROT" : projected_prices["CARROT"] * projected_harvest["CARROT"] / product_lifespan["CARROT"] if 30 - day > time_to_first_yield["CARROT"] else 0,
        "MELON" : projected_prices["MELON"] * projected_harvest["MELON"] / product_lifespan["MELON"] if 30 - day > time_to_first_yield["MELON"] else 0,
        "TOMATO" : projected_prices["TOMATO"] * projected_harvest["TOMATO"] / product_lifespan["TOMATO"] if 30 - day > time_to_first_yield["TOMATO"] else 0,
        "STRAWBERRY" : projected_prices["STRAWBERRY"] * projected_harvest["STRAWBERRY"] / product_lifespan["STRAWBERRY"] if 30 - day > time_to_first_yield["STRAWBERRY"] else 0,
        "EGG" : (projected_prices["EGG"] * projected_harvest["EGG"] / product_lifespan["EGG"] if 26 - day > 0 else 0) + (projected_prices["FERTILIZER"] * projected_harvest["FERTILIZER"] / product_lifespan["FERTILIZER"] if day < 30 else 0),
        "MILK" : (projected_prices["MILK"] * projected_harvest["MILK"] / product_lifespan["MILK"] if 22 - day > 0 else 0) + (projected_prices["FERTILIZER"] * projected_harvest["FERTILIZER"] / product_lifespan["FERTILIZER"] if day < 30 else 0),
        "WOOL" : (projected_prices["WOOL"] * projected_harvest["WOOL"] / product_lifespan["WOOL"] if 24 - day > 0 else 0) + (projected_prices["FERTILIZER"] * projected_harvest["FERTILIZER"] / product_lifespan["FERTILIZER"] if day < 30 else 0)
    }

    product_cost = {
        "WHEAT" : seed_cost["WHEAT"],
        "CARROT" : seed_cost["CARROT"],
        "MELON" : seed_cost["MELON"],
        "TOMATO" : seed_cost["TOMATO"] + prices["FERTILIZER"],
        "STRAWBERRY" : seed_cost["STRAWBERRY"] + prices["FERTILIZER"] * 2,
        "EGG" : seed_cost["EGG"] + (prices["WHEAT"] * (29 - day) if day < 29 else 0),
        "MILK" : seed_cost["MILK"] + (prices["WHEAT"] * (27 - day) if day < 27 else 0),
        "WOOL" : seed_cost["WOOL"] + (prices["WHEAT"] * (29 - day) if day < 29 else 0)
    }

    product_cost_per_day = {
        "WHEAT" : product_cost["WHEAT"] / product_lifespan["WHEAT"],
        "CARROT" : product_cost["CARROT"] / product_lifespan["CARROT"],
        "MELON" : product_cost["MELON"] / product_lifespan["MELON"],
        "TOMATO" : product_cost["TOMATO"] / product_lifespan["TOMATO"],
        "STRAWBERRY" : product_cost["STRAWBERRY"] / product_lifespan["STRAWBERRY"],
        "EGG" : product_cost["EGG"] / product_lifespan["EGG"] if day < 30 else product_cost["EGG"],
        "MILK" : product_cost["MILK"] / product_lifespan["MILK"] if day < 30 else product_cost["MILK"],
        "WOOL" : product_cost["WOOL"] / product_lifespan["WOOL"] if day < 30 else product_cost["WOOL"]
    }

    product_values = {
            "WHEAT" : product_revenue_per_day["WHEAT"] - product_cost_per_day["WHEAT"],
            "CARROT" : product_revenue_per_day["CARROT"] - product_cost_per_day["CARROT"],
            "MELON" : product_revenue_per_day["MELON"] - product_cost_per_day["MELON"],
            "TOMATO" : product_revenue_per_day["TOMATO"] - product_cost_per_day["TOMATO"],
            "STRAWBERRY" : product_revenue_per_day["STRAWBERRY"] - product_cost_per_day["STRAWBERRY"],
            "EGG" : product_revenue_per_day["EGG"] - product_cost_per_day["EGG"],
            "MILK" : product_revenue_per_day["MILK"] - product_cost_per_day["MILK"],
            "WOOL" : product_revenue_per_day["WOOL"] - product_cost_per_day["WOOL"]
    }
    #print('product_values:', product_values)

    remaining_seeds = seeds

    shed_contents = {
        "COW" : shed.get("COW", 0),
        "SHEEP" : shed.get("SHEEP", 0),
        "GOOSE" : shed.get("GOOSE", 0),
        "WHEAT" : shed.get("WHEAT", 0),
        "FERTILIZER" : shed.get("FERTILIZER", 0)
    }

    def get_closest_shed_tile(farmer_pos):
        fx, fy = farmer_pos
        closest_tile = None
        min_dist = float("inf")
        
        for sx, sy in shed_adjacent_tiles:
            dist = abs(fx - sx) + abs(fy - sy)
            if dist < min_dist:
                min_dist = dist
                closest_tile = (sx, sy)
                
        return closest_tile

    def update_value(product):
        projected_inventory[product] += projected_harvest[product] #note: if the opponent has a competent model, they will also buy this crop. It is possible that this will have to be weighted so that the expected inventory is increased greater than if only i bought this product
        projected_prices[product] = predict_price(projected_inventory[product], base_costs[product], T_values[product], below_funcs[product], below_targets[product], above_funcs[product], above_targets[product])
        product_revenue_per_day[product] = projected_prices[product] * projected_harvest[product] / product_lifespan[product] if product_lifespan[product] > 0 else 0
        product_values[product] = product_revenue_per_day[product] - product_cost_per_day[product]

    quads_allowed = 3
    # 1. State Extraction & Tracking
    # Parse coordinates, tile statuses, cash, and shop multipliers
    farmer_pos = my_farm["farmer"]
    hands_list = my_farm["hands"]
    worker_actions = []
    market_orders = []
    # 2. Market Execution (Hourly / Daily triggers)
    # Buy seeds, sell ready produce, hire farm hands
    if shed.get("FERTILIZER", 0) - sum(len(v) for v in fert_needed_in_col.values()) > 0:
        market_orders.append(["SELL", "FERTILIZER", shed.get("FERTILIZER", 0) - sum(len(v) for v in fert_needed_in_col.values())])
    #if shed.get("WHEAT", 0) > 0:
        #market_orders.append(["SELL", "WHEAT", 1])
    if shed.get("CARROT", 0) > 0:
        market_orders.append(["SELL", "CARROT", shed.get("CARROT", 0)])
    if shed.get("MELON", 0) > 0:
        market_orders.append(["SELL", "MELON", shed.get("MELON", 0)])
    if shed.get("TOMATO", 0) > 0:
        market_orders.append(["SELL", "TOMATO", shed.get("TOMATO", 0)])
    if shed.get("STRAWBERRY", 0) > 0:
        market_orders.append(["SELL", "STRAWBERRY", shed.get("STRAWBERRY", 0)])
    if shed.get("EGG", 0) > 0:
        market_orders.append(["SELL", "EGG", shed.get("EGG", 0)])
    if shed.get("MILK", 0) > 0:
        market_orders.append(["SELL", "MILK", shed.get("MILK", 0)])
    if shed.get("WOOL", 0) > 0:
        market_orders.append(["SELL", "WOOL", shed.get("WOOL", 0)])
    #if hour == 1:
        #print('product_values:', product_values)
        #print(projected_produced["MELON"])
    if hour == 0:
        #print('product_values:', product_values)
        # e.g., buy seeds or hire hands at the start of a new day
        #move this to hour 0 since theres prob less than 10 orders
        held_animals_rn = 0
        for i in range(len(hands_list) + 1):
            held_animals_rn += inventories[i].get("COW", 0) + inventories[i].get("SHEEP", 0) + inventories[i].get("GOOSE", 0)
        animals_in_shed_rn = shed.get("GOOSE", 0) + shed.get("COW", 0) + shed.get("SHEEP", 0)
        fertilizer_wanted = 0
        wheat_seed_wanted = 0
        carrot_seed_wanted = 0
        melon_seed_wanted = 0
        tomato_seed_wanted = 0
        strawberry_seed_wanted = 0
        goose_wanted = 0
        cow_wanted = 0
        sheep_wanted = 0
        wheat_wanted = 0
        total_seeds_and_animals = seeds.get("WHEAT", 0) + seeds.get("CARROT", 0) + seeds.get("MELON", 0) + seeds.get("TOMATO", 0) + seeds.get("STRAWBERRY", 0) + shed.get("GOOSE", 0) + shed.get("COW", 0) + shed.get("SHEEP", 0)
        start_money = max(0, my_farm["money"] - cost_of_n_farmers[5 * len(my_farm["unlocked_quadrants"])] * 2 - prices["WHEAT"] * (animals_my_farm + held_animals_rn + animals_in_shed_rn) * (2 if day == 0 else 1))
        unlocked_quads = len(my_farm["unlocked_quadrants"])
        affordable = True
        budget = max(0, int(start_money))
        chunks = build_chunks(
            list(product_values.keys()), seed_cost, prices, projected_harvest, projected_inventory,
            base_costs, T_values, below_funcs, below_targets, above_funcs, above_targets,
            product_lifespan, product_cost_per_day, budget, max_units=empty_tiles
        )
        bought, total_daily_profit = solve_allocation_np(chunks, budget, empty_tiles)

        wheat_seed_wanted = bought.get("WHEAT", 0) - seeds.get("WHEAT", 0)
        carrot_seed_wanted = bought.get("CARROT", 0) - seeds.get("CARROT", 0)
        melon_seed_wanted = bought.get("MELON", 0) - seeds.get("MELON", 0)
        tomato_seed_wanted = bought.get("TOMATO", 0) - seeds.get("TOMATO", 0)
        strawberry_seed_wanted = bought.get("STRAWBERRY", 0) - seeds.get("STRAWBERRY", 0)
        goose_wanted = bought.get("EGG", 0) - shed.get("GOOSE", 0)
        cow_wanted = bought.get("MILK", 0) - shed.get("COW", 0)
        sheep_wanted = bought.get("WOOL", 0) - shed.get("SHEEP", 0)

        if goose_wanted > 0:
            market_orders.append(["BUY_ANIMAL", "GOOSE", goose_wanted])
        if cow_wanted > 0:
            market_orders.append(["BUY_ANIMAL", "COW", cow_wanted])
        if sheep_wanted > 0:
            market_orders.append(["BUY_ANIMAL", "SHEEP", sheep_wanted])
        if goose_wanted + cow_wanted + sheep_wanted > 0:
            wheat_wanted += goose_wanted + cow_wanted + sheep_wanted
        total_animals = sum(animals_in_col.values())
        animals_in_shed = shed.get("GOOSE", 0) + shed.get("COW", 0) + shed.get("SHEEP", 0)
        if total_animals + wheat_wanted + animals_in_shed > shed_contents["WHEAT"] and day < 29:
            market_orders.append(["BUY_PRODUCT", "WHEAT", total_animals + wheat_wanted + animals_in_shed - shed_contents["WHEAT"]])
        if wheat_seed_wanted > 0:
            market_orders.append(["BUY_SEED", "WHEAT", wheat_seed_wanted])
        if carrot_seed_wanted > 0:
            market_orders.append(["BUY_SEED", "CARROT", carrot_seed_wanted])
        if melon_seed_wanted > 0:
            market_orders.append(["BUY_SEED", "MELON", melon_seed_wanted])
        if tomato_seed_wanted > 0:
            market_orders.append(["BUY_SEED", "TOMATO", tomato_seed_wanted])
        if strawberry_seed_wanted > 0:
            market_orders.append(["BUY_SEED", "STRAWBERRY", strawberry_seed_wanted])
        if sum(len(v) for v in fert_needed_in_col.values()) - shed.get("FERTILIZER", 0) > 0:
            market_orders.append(["BUY_PRODUCT", "FERTILIZER", sum(len(v) for v in fert_needed_in_col.values()) - shed.get("FERTILIZER", 0)])
        #if tomato_seed_wanted + strawberry_seed_wanted * 2 > 0:
        #    market_orders.append(["BUY_PRODUCT", "FERTILIZER", tomato_seed_wanted + strawberry_seed_wanted * 2])
        #if day == 0:
            #product_values["MELON"] += 100
            #if my_farm["money"] >= 200 and seeds.get("MELON", 0) < 5:
            #    market_orders.append(["BUY_SEED", "MELON", 5])
        if day == 29:
            for _ in range(len(my_farm["unlocked_quadrants"]) * 4 - my_farm.get("hires_today", 0)):
                market_orders.append(["HIRE"])
            worker_actions.append(["HARVEST"])
        else:
            for i in range(len(my_farm["unlocked_quadrants"]) * 5 - 1 - my_farm.get("hires_today", 0)):
                market_orders.append(["HIRE"])

        #market_orders.append(["BUY_SEED", "WHEAT", 10])
        current_x, current_y = farmer_pos
        if current_x != 0:
            worker_actions.append(["PASS"]) #why not move
        #print(market_orders)

    # 3. Task Selection & Route Execution
    # Determine what the farmer standing on (farmer_pos[0], farmer_pos[1]) needs to do
    elif hour < 24:
        if day == 29:
            for _ in range(len(my_farm["unlocked_quadrants"]) * 4 - my_farm.get("hires_today", 0)):
                market_orders.append(["HIRE"])
            if shed.get("FERTILIZER", 0) > 0:
                market_orders.append(["SELL", "FERTILIZER", shed.get("FERTILIZER")])
            HOURS_IN_DAY = 24
            hours_left = (HOURS_IN_DAY - 1) - hour
            total_workers = len(hands_list) + 1
            if shed.get("WHEAT", 0) > 0:
                market_orders.append(["SELL", "WHEAT", shed.get("WHEAT")])
            # 1. Normalize shed adjacent tiles for set lookup
            shed_adjacent_set = {tuple(t) for t in shed_adjacent_tiles}

            # 2. Shared mutable pool of harvestable crops for this tick (normalized to tuples)
            unclaimed_crops = [tuple(c) for c in harvestable]

            # 3. Process each worker sequentially
            for i in range(total_workers):
                current_pos = tuple(farmer_pos if i == 0 else hands_list[i - 1])
                closest_drop = tuple(get_closest_shed_tile(current_pos))
                dist_to_shed = abs(current_pos[0] - closest_drop[0]) + abs(current_pos[1] - closest_drop[1])

                # Hard curfew: walk distance + 1 turn to drop + 1 buffer
                must_return = hours_left <= (dist_to_shed + 3)
                if i == 0 and hour < 8:
                    worker_actions.append(step_toward(current_pos, (0, 0)))
                elif i == 1 and hour < 8:
                    worker_actions.append(step_toward(current_pos, (9, 0)))
                else:
                    # Check if this worker can claim a crop
                    chosen_crop = None
                    if not must_return and len(unclaimed_crops) > 0:
                        # Find the crop closest to THIS worker's current coordinates
                        chosen_crop = min(
                            unclaimed_crops,
                            key=lambda c: abs(c[0] - current_pos[0]) + abs(c[1] - current_pos[1])
                        )
                        # Remove immediately so subsequent workers in this loop cannot pick it
                        unclaimed_crops.remove(chosen_crop)

                    # --- EXECUTE ACTIONS ---
                    # Case A: Return to shed (curfew reached OR no crops left to claim)
                    if chosen_crop is None:
                        if current_pos not in shed_adjacent_set:
                            worker_actions.append(step_toward(current_pos, closest_drop))
                        else:
                            worker_actions.append(["DROP"])

                    # Case B: Go to or harvest the claimed crop
                    else:
                        if current_pos == chosen_crop:
                            worker_actions.append(["HARVEST"])
                            # Remove from global harvestable list since it is harvested this tick
                            if chosen_crop in harvestable:
                                harvestable.remove(chosen_crop)
                            elif list(chosen_crop) in harvestable:
                                harvestable.remove(list(chosen_crop))
                        else:
                            move = step_toward(current_pos, chosen_crop)
                            worker_actions.append(move)


        else:
            for i in range(len(my_farm["unlocked_quadrants"]) * 5 - 1 - my_farm.get("hires_today", 0)):
                market_orders.append(["HIRE"])
            held_animals_rn = 0
            for i in range(len(hands_list) + 1):
                held_animals_rn += inventories[i].get("COW", 0) + inventories[i].get("SHEEP", 0) + inventories[i].get("GOOSE", 0)
            animals_in_shed_rn = shed.get("GOOSE", 0) + shed.get("COW", 0) + shed.get("SHEEP", 0)
            if shed.get("WHEAT", 0) - sum(animals_in_col.values()) - held_animals_rn - animals_in_shed_rn > 0:
                market_orders.append(["SELL", "WHEAT", shed.get("WHEAT", 0) - sum(animals_in_col.values()) - held_animals_rn - animals_in_shed_rn])
            wheat_seed_wanted = 0
            carrot_seed_wanted = 0
            melon_seed_wanted = 0
            tomato_seed_wanted = 0
            strawberry_seed_wanted = 0
            total_seeds_and_animals = seeds.get("WHEAT", 0) + seeds.get("CARROT", 0) + seeds.get("MELON", 0) + seeds.get("TOMATO", 0) + seeds.get("STRAWBERRY", 0) + shed.get("GOOSE", 0) + shed.get("COW", 0) + shed.get("SHEEP", 0) + sum(inventories[k].get("COW", 0) + inventories[k].get("SHEEP", 0) + inventories[k].get("GOOSE", 0) for k in range(len(hands_list) + 1)) 
            start_money = max(0, my_farm["money"] - cost_of_n_farmers[5 * len(my_farm["unlocked_quadrants"])] * 2 - prices["WHEAT"] * (animals_my_farm + held_animals_rn + animals_in_shed_rn) * (2 if day == 0 else 0))
            unlocked_quads = len(my_farm["unlocked_quadrants"])
            affordable = True
            budget = max(0, int(start_money))
            chunks = build_chunks(
                list(product_values.keys()), seed_cost, prices, projected_harvest, projected_inventory,
                base_costs, T_values, below_funcs, below_targets, above_funcs, above_targets,
                product_lifespan, product_cost_per_day, budget, max_units=empty_tiles
            )
            bought, total_daily_profit = solve_allocation_np(chunks, budget, empty_tiles)
    
            wheat_seed_wanted = bought.get("WHEAT", 0) - seeds.get("WHEAT", 0)
            carrot_seed_wanted = bought.get("CARROT", 0) - seeds.get("CARROT", 0)
            melon_seed_wanted = bought.get("MELON", 0) - seeds.get("MELON", 0)
            tomato_seed_wanted = bought.get("TOMATO", 0) - seeds.get("TOMATO", 0)
            strawberry_seed_wanted = bought.get("STRAWBERRY", 0) - seeds.get("STRAWBERRY", 0)
            goose_wanted = bought.get("EGG", 0) - shed.get("GOOSE", 0)
            cow_wanted = bought.get("MILK", 0) - shed.get("COW", 0)
            sheep_wanted = bought.get("WOOL", 0) - shed.get("SHEEP", 0)

            if wheat_seed_wanted > 0:
                market_orders.append(["BUY_SEED", "WHEAT", wheat_seed_wanted])
            if carrot_seed_wanted > 0:
                market_orders.append(["BUY_SEED", "CARROT", carrot_seed_wanted])
            if melon_seed_wanted > 0:
                market_orders.append(["BUY_SEED", "MELON", melon_seed_wanted])
            if tomato_seed_wanted > 0:
                market_orders.append(["BUY_SEED", "TOMATO", tomato_seed_wanted])
            if strawberry_seed_wanted > 0:
                market_orders.append(["BUY_SEED", "STRAWBERRY", strawberry_seed_wanted])
            animals_in_shed = shed.get("GOOSE", 0) + shed.get("COW", 0) + shed.get("SHEEP", 0)
            animals_needed = defaultdict(int)
            goose_needed = defaultdict(int)
            cow_or_sheep_needed = defaultdict(int)
            for (x,y) in empty_pasture_list:
                if y <= 4:
                    cow_or_sheep_needed[(x, 0)] += 1
                else:
                    cow_or_sheep_needed[(x, 9)] += 1
                animals_in_shed -= 1
            for (x,y) in empty_coop_list:
                if y <= 4:
                    goose_needed[(x, 0)] += 1
                else:
                    goose_needed[(x, 9)] += 1
                animals_in_shed -= 1
            while empty_tile_list and animals_in_shed > 0:
                col, row = (-2,-2)
                for (x,y) in empty_tile_list:
                    if abs(4 - y) < abs(4 - row):
                        col = x
                        row = y
                if row <= 4:
                    animals_needed[(col, 0)] += 1
                else:
                    animals_needed[(col, 9)] += 1
                animals_in_shed -= 1
                empty_tile_list.remove((col, row))
            
            for i in range(len(hands_list) + 1):
                held_animals = inventories[i].get("COW", 0) + inventories[i].get("SHEEP", 0) + inventories[i].get("GOOSE", 0)
                if i == 0:
                    current_x, current_y = farmer_pos
                else:
                    current_x, current_y = hands_list[i - 1]
                current_tile = my_tiles[current_y][current_x]
                for j in range(len(hands_list) + 1):
                    if goose_needed[farmer_col[j]] - inventories[j].get("GOOSE", 0) <= 0:
                        animals_needed[farmer_col[j]] -= inventories[j].get("GOOSE", 0) - goose_needed[farmer_col[j]]
                        goose_needed[farmer_col[j]] = 0
                    else:
                        goose_needed[farmer_col[j]] -= inventories[j].get("GOOSE", 0)
                    if cow_or_sheep_needed[farmer_col[j]] - inventories[j].get("COW", 0) - inventories[j].get("SHEEP", 0) <= 0:
                        animals_needed[farmer_col[j]] -= inventories[j].get("COW", 0) + inventories[j].get("SHEEP", 0) - cow_or_sheep_needed[farmer_col[j]]
                        cow_or_sheep_needed[farmer_col[j]] = 0
                    else:
                        cow_or_sheep_needed[farmer_col[j]] -= inventories[j].get("COW", 0) + inventories[j].get("SHEEP", 0)
                desired_col, r = farmer_col[i]
                if current_x != desired_col:
                    g_needed = goose_needed[farmer_col[i]]
                    goose_needed[farmer_col[i]] = 0
                    c_s_needed = cow_or_sheep_needed[farmer_col[i]]
                    cow_or_sheep_needed[farmer_col[i]] = 0
                    other = animals_needed[farmer_col[i]]
                    animals_needed[farmer_col[i]] = 0
                    if shed_contents["GOOSE"] > 0 and g_needed > 0 and (current_x, current_y) in shed_adjacent_tiles:
                        worker_actions.append(["PICKUP", "GOOSE", min(shed_contents["GOOSE"] - sum(goose_needed.values()), g_needed + other)])
                        shed_contents["GOOSE"] -= min(shed_contents["GOOSE"] - sum(goose_needed.values()), g_needed + other)
                    elif shed_contents["COW"] > 0 and c_s_needed > 0 and (current_x, current_y) in shed_adjacent_tiles:
                        worker_actions.append(["PICKUP", "COW", min(shed_contents["COW"] - sum(cow_or_sheep_needed.values()), c_s_needed + other)])
                        shed_contents["COW"] -= min(shed_contents["COW"] - sum(cow_or_sheep_needed.values()), c_s_needed + other)
                    elif shed_contents["SHEEP"] > 0 and c_s_needed > 0 and (current_x, current_y) in shed_adjacent_tiles:
                        worker_actions.append(["PICKUP", "SHEEP", min(shed_contents["SHEEP"] - sum(cow_or_sheep_needed.values()), c_s_needed + other)])
                        shed_contents["SHEEP"] -= min(shed_contents["SHEEP"] - sum(cow_or_sheep_needed.values()), c_s_needed + other)
                    elif other > 0 and (current_x, current_y) in shed_adjacent_tiles:
                        if shed_contents["GOOSE"] - sum(goose_needed.values()) >= other:
                            worker_actions.append(["PICKUP", "GOOSE", min(shed_contents["GOOSE"] - sum(goose_needed.values()), other)])
                            shed_contents["GOOSE"] -= min(shed_contents["GOOSE"] - sum(goose_needed.values()), other)
                        elif shed_contents["COW"] - sum(cow_or_sheep_needed.values()) >= other:
                            worker_actions.append(["PICKUP", "COW", min(shed_contents["COW"] - sum(cow_or_sheep_needed.values()), other)])
                            shed_contents["COW"] -= min(shed_contents["COW"] - sum(cow_or_sheep_needed.values()), other)
                        elif shed_contents["SHEEP"] - sum(cow_or_sheep_needed.values()) >= other:
                            worker_actions.append(["PICKUP", "SHEEP", min(shed_contents["SHEEP"] - sum(cow_or_sheep_needed.values()), other)])
                            shed_contents["SHEEP"] -= min(shed_contents["SHEEP"] - sum(cow_or_sheep_needed.values()), other)
                        elif shed_contents["GOOSE"] - sum(goose_needed.values()) > 0:
                            worker_actions.append(["PICKUP", "GOOSE", min(shed_contents["GOOSE"] - sum(goose_needed.values()), other)])
                            shed_contents["GOOSE"] -= min(shed_contents["GOOSE"] - sum(goose_needed.values()), other)
                        elif shed_contents["COW"] - sum(cow_or_sheep_needed.values()) >= 0:
                            worker_actions.append(["PICKUP", "COW", min(shed_contents["COW"] - sum(cow_or_sheep_needed.values()), other)])
                            shed_contents["COW"] -= min(shed_contents["COW"] - sum(cow_or_sheep_needed.values()), other)
                        elif shed_contents["SHEEP"] - sum(cow_or_sheep_needed.values()) >= 0:
                            worker_actions.append(["PICKUP", "SHEEP", min(shed_contents["SHEEP"] - sum(cow_or_sheep_needed.values()), other)])
                            shed_contents["SHEEP"] -= min(shed_contents["SHEEP"] - sum(cow_or_sheep_needed.values()), other)
                        elif shed_contents["COW"] + shed_contents["SHEEP"] - sum(cow_or_sheep_needed.values()) >= 0:
                            if shed_contents["COW"] > shed_contents["SHEEP"]:
                                    worker_actions.append(["PICKUP", "COW", min(shed_contents["COW"], other)])
                                    shed_contents["COW"] -= min(shed_contents["COW"], other)
                            else:
                                worker_actions.append(["PICKUP", "SHEEP", min(shed_contents["SHEEP"], other)])
                                shed_contents["SHEEP"] -= min(shed_contents["SHEEP"], other)
                        else:
                            worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                    elif (current_x, current_y) in shed_adjacent_tiles and inventories[i].get("WHEAT", 0) < held_animals + animals_in_col[farmer_col[i]] and shed.get("WHEAT", 0) > 0:
                        worker_actions.append(["PICKUP", "WHEAT", min(shed.get("WHEAT", 0), held_animals + animals_in_col[farmer_col[i]] - inventories[i].get("WHEAT", 0))])
                    elif len(fert_needed_in_col[farmer_col[i]]) - inventories[i].get("FERTILIZER", 0) > 0 and (current_x, current_y) in shed_adjacent_tiles and shed.get("FERTILIZER", 0) > 0:
                        worker_actions.append(["PICKUP", "FERTILIZER", min(shed.get("FERTILIZER", 0), len(fert_needed_in_col[farmer_col[i]]) - inventories[i].get("FERTILIZER", 0))])
                        print('pick up stix')
                    else: 
                        worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                    '''
                    if shed.get("GOOSE", 0) > 0 and empty_tiles_in_col[farmer_col[i]] + empty_coops_in_col[farmer_col[i]] > 0 and (current_x, current_y) in shed_adjacent_tiles:
                        worker_actions.append(["PICKUP", "GOOSE", min(2, empty_tiles_in_col[farmer_col[i]] + empty_coops_in_col[farmer_col[i]], shed.get("GOOSE", 0))])
                        held_animals = min(math.ceil(animals_in_shed // 5), empty_tiles_in_col[farmer_col[i]] + empty_coops_in_col[farmer_col[i]], shed.get("GOOSE", 0))
                    elif shed.get("COW", 0) > 0 and empty_tiles_in_col[farmer_col[i]] + empty_pastures_in_col[farmer_col[i]] > 0 and (current_x, current_y) in shed_adjacent_tiles:
                        worker_actions.append(["PICKUP", "COW", min(2, empty_tiles_in_col[farmer_col[i]] + empty_pastures_in_col[farmer_col[i]], shed.get("COW", 0))])
                        held_animals = min(math.ceil(animals_in_shed // 5), empty_tiles_in_col[farmer_col[i]]  + empty_pastures_in_col[farmer_col[i]], shed.get("COW", 0))
                    elif shed.get("SHEEP", 0) > 0 and empty_tiles_in_col[farmer_col[i]] + empty_pastures_in_col[farmer_col[i]] > 0 and (current_x, current_y) in shed_adjacent_tiles:
                        worker_actions.append(["PICKUP", "SHEEP", min(2, empty_tiles_in_col[farmer_col[i]]  + empty_pastures_in_col[farmer_col[i]], shed.get("SHEEP", 0))])
                        held_animals = min(math.ceil(animals_in_shed // 5), empty_tiles_in_col[farmer_col[i]]  + empty_pastures_in_col[farmer_col[i]], shed.get("SHEEP", 0))
                        '''
                else:
                    if current_tile == None:
                        if current_x + current_y >= 6:
                            if remaining_seeds.get("MELON", 0) > 0:
                                worker_actions.append(["PLANT", "MELON"])
                                remaining_seeds["MELON"] -= 1
                            elif "COW" in inventories[i] or "SHEEP" in inventories[i]:
                                worker_actions.append(["BUILD_PASTURE"])
                            elif "GOOSE" in inventories[i]:
                                worker_actions.append(["BUILD_COOP"])
                            elif remaining_seeds.get("WHEAT", 0) > 0:
                                worker_actions.append(["PLANT", "WHEAT"])
                                remaining_seeds["WHEAT"] -= 1
                            elif remaining_seeds.get("CARROT", 0) > 0:
                                worker_actions.append(["PLANT", "CARROT"])
                                remaining_seeds["CARROT"] -= 1
                            elif remaining_seeds.get("TOMATO", 0) > 0:
                                worker_actions.append(["PLANT", "TOMATO"])
                                remaining_seeds["TOMATO"] -= 1
                            elif remaining_seeds.get("STRAWBERRY", 0) > 0:
                                worker_actions.append(["PLANT", "STRAWBERRY"])
                                remaining_seeds["STRAWBERRY"] -= 1
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                        else:
                            if "COW" in inventories[i] or "SHEEP" in inventories[i]:
                                worker_actions.append(["BUILD_PASTURE"])
                            elif "GOOSE" in inventories[i]:
                                worker_actions.append(["BUILD_COOP"])
                            elif remaining_seeds.get("WHEAT", 0) > 0:
                                worker_actions.append(["PLANT", "WHEAT"])
                                remaining_seeds["WHEAT"] -= 1
                            elif remaining_seeds.get("CARROT", 0) > 0:
                                worker_actions.append(["PLANT", "CARROT"])
                                remaining_seeds["CARROT"] -= 1
                            elif remaining_seeds.get("TOMATO", 0) > 0:
                                worker_actions.append(["PLANT", "TOMATO"])
                                remaining_seeds["TOMATO"] -= 1
                            elif remaining_seeds.get("STRAWBERRY", 0) > 0:
                                worker_actions.append(["PLANT", "STRAWBERRY"])
                                remaining_seeds["STRAWBERRY"] -= 1
                            elif remaining_seeds.get("MELON", 0) > 0:
                                worker_actions.append(["PLANT", "MELON"])
                                remaining_seeds["MELON"] -= 1
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                    elif current_tile == "LOCKED":
                        if current_y != r:
                            worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                        else:
                            worker_actions.append(["PASS"])
                    elif current_tile.get("kind") == "PLANT":
                        age = day - current_tile.get("planted_day")
                        lifespan = product_lifespan[current_tile.get("crop")]
                        first_yield_day = time_to_first_yield[current_tile.get("crop")]
                        bonus_day_start = math.ceil(lifespan / 2)
                        max_harvest_size = max_harvest[current_tile.get("crop")]
                        crop_price = prices[current_tile.get("crop")]
                        if current_tile.get("consecutive_unwatered") > 0 and not current_tile.get("watered_today"):
                            worker_actions.append(["WATER"])
                        elif current_tile.get("crop") == "WHEAT":
                            if age >= bonus_day_start: #bonus window of wheat
                                if (current_tile.get("yield_units") >= max_harvest_size and age >= first_yield_day):
                                    worker_actions.append(["HARVEST"])
                                elif not current_tile.get("watered_today"):
                                    worker_actions.append(["WATER"])
                                elif age == bonus_day_start and current_tile.get("fertilized_until_day") == -1 and inventories[i].get("FERTILIZER", 0) > 0 and prices["FERTILIZER"] < 2 * crop_price:
                                    worker_actions.append(["FERTILIZE"])
                                    #print('wheat fertilize')
                                elif age >= lifespan or (current_tile.get("yield_units") >= max_harvest_size and age >= first_yield_day):
                                    worker_actions.append(["HARVEST"])
                                else:
                                    if current_y != r:
                                        worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                    else:
                                        worker_actions.append(["PASS"])
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                        elif current_tile.get("crop") == "CARROT":
                            if age >= bonus_day_start: #bonus window of carrot
                                if (current_tile.get("yield_units") >= max_harvest_size and age >= first_yield_day):
                                    worker_actions.append(["HARVEST"])
                                elif not current_tile.get("watered_today"):
                                    worker_actions.append(["WATER"])
                                elif age == bonus_day_start and current_tile.get("fertilized_until_day") == -1 and inventories[i].get("FERTILIZER", 0) > 0 and prices["FERTILIZER"] < crop_price:
                                    worker_actions.append(["FERTILIZE"]) 
                                    #print('carrot fertilize')
                                elif age >= lifespan or (current_tile.get("yield_units") >= max_harvest_size and age >= first_yield_day):
                                    worker_actions.append(["HARVEST"])
                                else:
                                    if current_y != r:
                                        worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                    else:
                                        worker_actions.append(["PASS"])
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                        elif current_tile.get("crop") == "MELON":
                            if age >= bonus_day_start: #bonus window of melon
                                if (current_tile.get("yield_units") >= max_harvest_size and age >= first_yield_day):
                                    worker_actions.append(["HARVEST"])
                                elif not current_tile.get("watered_today"):
                                    worker_actions.append(["WATER"])
                                #elif age == bonus_day_start and current_tile.get("fertilized_until_day") == -1 and inventories[i].get("FERTILIZER", 0) > 0 and prices["FERTILIZER"] < crop_price:
                                    #worker_actions.append(["FERTILIZE"])
                                elif age >= lifespan or (current_tile.get("yield_units") >= max_harvest_size and age >= first_yield_day):
                                    worker_actions.append(["HARVEST"])
                                else:
                                    if current_y != r:
                                        worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                    else:
                                        worker_actions.append(["PASS"])
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                        elif current_tile.get("crop") == "TOMATO":
                            if age in {7, 8, 9, 10, 11}: #bonus window of tomato
                                if (age >= lifespan and current_tile.get("yield_units") > 0) or (current_tile.get("yield_units") >= max_harvest_size - 1 and age >= first_yield_day):
                                    worker_actions.append(["HARVEST"])
                                elif not current_tile.get("watered_today") and age < 11:
                                    worker_actions.append(["WATER"])
                                elif (age == 7 or age == 8) and current_tile.get("fertilized_until_day") == -1 and inventories[i].get("FERTILIZER", 0) > 0 and prices["FERTILIZER"] < crop_price * 3:
                                    worker_actions.append(["FERTILIZE"]) 
                                    #print('tomato fertilize')
                                elif age >= lifespan:
                                    worker_actions.append(["DIG"])
                                else:
                                    if current_y != r:
                                        worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                    else:
                                        worker_actions.append(["PASS"])
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                        elif current_tile.get("crop") == "STRAWBERRY":
                            if age in {9, 10, 11, 12, 13, 14, 15, 16}: #bonus window of strawberry
                                if (age >= lifespan and current_tile.get("yield_units") > 0) or (current_tile.get("yield_units") >= max_harvest_size - 1 and age >= first_yield_day):
                                    worker_actions.append(["HARVEST"])
                                elif not current_tile.get("watered_today") and age < lifespan:
                                    worker_actions.append(["WATER"])
                                elif age >= 9 and (current_tile.get("fertilized_until_day") == -1  or current_tile.get("fertilized_until_day") < day) and inventories[i].get("FERTILIZER", 0) > 0 and prices["FERTILIZER"] < 2 * crop_price:
                                    worker_actions.append(["FERTILIZE"]) 
                                    #print('strawberry fertilize')
                                elif age >= lifespan:
                                    worker_actions.append(["DIG"])
                                else:
                                    if current_y != r:
                                        worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                    else:
                                        worker_actions.append(["PASS"])
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                        else:
                            print('unknown plant detected')
                    elif current_tile.get("kind") == "COOP" or current_tile.get("kind") == "PASTURE":
                        age = day - current_tile.get("placed_day", 0)

                        if current_tile.get("animal") == None and current_tile.get("kind") == "PASTURE":
                            if inventories[i].get("COW", 0) > 0:
                                worker_actions.append(["PLACE", "COW"])
                            elif inventories[i].get("SHEEP", 0) > 0:
                                worker_actions.append(["PLACE", "SHEEP"])
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                        elif current_tile.get("animal") == None and current_tile.get("kind") == "COOP":
                            if inventories[i].get("GOOSE", 0) > 0:
                                worker_actions.append(["PLACE", "GOOSE"])
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                        elif (current_tile.get("consecutive_unfed") == 1 or current_tile.get("cared_today") or age == time_to_first_yield[current_tile.get("animal")] - 1) and not current_tile.get("fed_today") and inventories[i].get("WHEAT", 0) > 0:
                            worker_actions.append(["FEED"])
                        elif current_tile.get("fertilizer_available"):
                            worker_actions.append(["COLLECT_FERTILIZER"])
                        elif current_tile.get("animal") == "GOOSE":
                            if current_tile.get("yield_units") >= 4:
                                worker_actions.append(["HARVEST"])
                            elif (current_tile.get("pending_care_bonus") < 3 or current_tile.get("fed_today")) and not current_tile.get("cared_today"):
                                worker_actions.append(["CARE"])
                            elif current_tile.get("yield_units") > 0:
                                worker_actions.append(["HARVEST"])
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                        elif current_tile.get("animal") == "COW":
                            if current_tile.get("yield_units") >= 6:
                                worker_actions.append(["HARVEST"])
                            elif (current_tile.get("pending_care_bonus") < 5 or current_tile.get("fed_today")) and not current_tile.get("cared_today"):
                                worker_actions.append(["CARE"])
                            elif current_tile.get("yield_units") > 0:
                                worker_actions.append(["HARVEST"])
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                        elif current_tile.get("animal") == "SHEEP":
                            if current_tile.get("yield_units") >= 6:
                                worker_actions.append(["HARVEST"])
                            elif (current_tile.get("pending_care_bonus") < 5 or current_tile.get("fed_today")) and not current_tile.get("cared_today"):
                                worker_actions.append(["CARE"])
                            elif current_tile.get("yield_units") > 0:
                                worker_actions.append(["HARVEST"])
                            else:
                                if current_y != r:
                                    worker_actions.append(step_toward((current_x, current_y), farmer_col[i]))
                                else:
                                    worker_actions.append(["PASS"])
                    elif current_tile.get("kind") == "WEED":
                        worker_actions.append(["DIG"])
            


    '''if current_tile and current_tile.get("kind") == "PLANT":
        if current_tile.get("yield_units", 0) > 0:
            worker_actions.append("HARVEST")
        elif not current_tile.get("watered_today"):
            worker_actions.append("WATER")
        else:
            worker_actions.append("EAST")  # Move to next tile
    else:
        # Default movement or maintenance path
        worker_actions.append("PASS")
        '''

    # 4. Return commands to the engine
    # (Matches the competition's submission API schema)
    #print(worker_actions)
    return {
        "farmer": worker_actions[0] if worker_actions else "PASS",
        "hands": [] if len(worker_actions) < 2  else worker_actions[1:],
        "market": market_orders,
    }