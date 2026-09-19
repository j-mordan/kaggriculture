import math


def step_toward(pos, target):
    """Return a single movement action (NORTH/SOUTH/EAST/WEST) that reduces
    Manhattan distance to target. Locked tiles are passable, so no
    obstacle-avoidance is needed."""
    px, py = pos
    tx, ty = target
    if px == tx and py == ty:
        return None
    dx, dy = tx - px, ty - py
    # Prefer whichever axis has the larger gap.
    if abs(dx) >= abs(dy) and dx != 0:
        return ["EAST"] if dx > 0 else ["WEST"]
    if dy != 0:
        return ["SOUTH"] if dy > 0 else ["NORTH"]
    return None

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

    daily_demand = { #default
        "WHEAT" : 1,
        "CARROT" : 1,
        "MELON" : 1,
        "TOMATO" : 1,
        "STRAWBERRY" : 1,
        "EGG" : 1,
        "MILK" : 1,
        "WOOL" : 1
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
        "WOOL" : 1.5
    }

    projected_produced = {
        "WHEAT" : 0,
        "CARROT" : 0,
        "MELON" : 0,
        "TOMATO" : 0,
        "STRAWBERRY" : 0,
        "EGG" : 0,
        "MILK" : 0,
        "WOOL" : 0
    }

    max_harvest = {
        "WHEAT" : 6,
        "CARROT" : 4,
        "MELON" : 6,
        "TOMATO" : 12,
        "STRAWBERRY" : 12,
        "EGG" : 30 - day, #could need to be changed, this neglects the time for goose to mature
        "MILK" : (30 - day) // 2, #could need to be changed
        "WOOL" : (30 - day) // 3 #could need to be changed
    }

    empty_tiles = 0
    for tile_row in my_tiles:
        for tile in tile_row:
            if tile == None:
                empty_tiles += 1
                continue
            elif tile == "LOCKED":
                continue
            elif tile.get("kind") == "PLANT":
                if tile.get("crop") == "WHEAT":
                    projected_produced["WHEAT"] += max_harvest["WHEAT"]
                elif tile.get("crop") == "CARROT":
                    projected_produced["CARROT"] += max_harvest["CARROT"]
                elif tile.get("crop") == "MELON":
                    projected_produced["MELON"] += max_harvest["MELON"]
                elif tile.get("crop") == "TOMATO":
                    projected_produced["TOMATO"] += max_harvest["TOMATO"]
                elif tile.get("crop") == "STRAWBERRY":
                    projected_produced["STRAWBERRY"] += max_harvest["STRAWBERRY"]
            elif tile.get("kind") == "WEED":
                continue
            elif tile.get("kind") == "COOP":
                if tile.get("animal") == "GOOSE":
                    projected_produced["EGG"] += max_harvest["EGG"] - ((day - tile.get("placed_day")) if day - tile.get("placed_day") < 4 else 0)
            elif tile.get("kind") == "PASTURE":
                if tile.get("animal") == "COW":
                    projected_produced["MILK"] += max_harvest["MILK"] - ((day - tile.get("placed_day")) // 2 if day - tile.get("placed_day") < 8 else 0)
                elif tile.get("animal") == "SHEEP":
                    projected_produced["WOOL"] += max_harvest["WOOL"] - ((day - tile.get("placed_day")) // 3 if day - tile.get("placed_day") < 6 else 0)
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
                    projected_produced["WHEAT"] += max_harvest["WHEAT"]
                elif tile.get("crop") == "CARROT":
                    projected_produced["CARROT"] += max_harvest["CARROT"]
                elif tile.get("crop") == "MELON":
                    projected_produced["MELON"] += max_harvest["MELON"]
                elif tile.get("crop") == "TOMATO":
                    projected_produced["TOMATO"] += max_harvest["TOMATO"]
                elif tile.get("crop") == "STRAWBERRY":
                    projected_produced["STRAWBERRY"] += max_harvest["STRAWBERRY"]
            elif tile.get("kind") == "WEED":
                continue
            elif tile.get("kind") == "COOP":
                if tile.get("animal") == "GOOSE":
                    projected_produced["EGG"] += max_harvest["EGG"] - ((day - tile.get("placed_day")) if day - tile.get("placed_day") < 4 else 0)
            elif tile.get("kind") == "PASTURE":
                if tile.get("animal") == "COW":
                    projected_produced["MILK"] += max_harvest["MILK"] - ((day - tile.get("placed_day")) // 2 if day - tile.get("placed_day") < 8 else 0)
                elif tile.get("animal") == "SHEEP":
                    projected_produced["WOOL"] += max_harvest["WOOL"] - ((day - tile.get("placed_day")) // 3 if day - tile.get("placed_day") < 6 else 0)
            else:
                print('unexpected opp farm tile kind:', tile.get("kind"))
                continue

    current_demand_minus_projected_supply = {
        "WHEAT" : daily_demand["WHEAT"] - projected_produced["WHEAT"],
        "CARROT" : daily_demand["CARROT"] - projected_produced["CARROT"],
        "MELON" : daily_demand["MELON"] - projected_produced["MELON"],
        "TOMATO" : daily_demand["TOMATO"] - projected_produced["TOMATO"],
        "STRAWBERRY" : daily_demand["STRAWBERRY"] - projected_produced["STRAWBERRY"],
        "EGG" : daily_demand["EGG"] - projected_produced["EGG"],
        "MILK" : daily_demand["MILK"] - projected_produced["MILK"],
        "WOOL" : daily_demand["WOOL"] - projected_produced["WOOL"]
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
    ) / len(remaining_wool_days) if remaining_wool_days else 0
    }

    total_demand_minus_supply = {
        "WHEAT" : current_demand_minus_projected_supply["WHEAT"] + projected_demand_increase["WHEAT"],
        "CARROT" : current_demand_minus_projected_supply["CARROT"] + projected_demand_increase["CARROT"],
        "MELON" : current_demand_minus_projected_supply["MELON"] + projected_demand_increase["MELON"],
        "TOMATO" : current_demand_minus_projected_supply["TOMATO"] + projected_demand_increase["TOMATO"],
        "STRAWBERRY" : current_demand_minus_projected_supply["STRAWBERRY"] + projected_demand_increase["STRAWBERRY"],
        "EGG" : current_demand_minus_projected_supply["EGG"] + projected_demand_increase["EGG"],
        "MILK" : current_demand_minus_projected_supply["MILK"] + projected_demand_increase["MILK"],
        "WOOL" : current_demand_minus_projected_supply["WOOL"] + projected_demand_increase["WOOL"]
    }

    projected_inventory = {
        "WHEAT" : obs["market"]["inventory"].get("WHEAT", 0) - total_demand_minus_supply["WHEAT"],
        "CARROT" : obs["market"]["inventory"].get("CARROT", 0) - total_demand_minus_supply["CARROT"],
        "MELON" : obs["market"]["inventory"].get("MELON", 0) - total_demand_minus_supply["MELON"],
        "TOMATO" : obs["market"]["inventory"].get("TOMATO", 0) - total_demand_minus_supply["TOMATO"],
        "STRAWBERRY" : obs["market"]["inventory"].get("STRAWBERRY", 0) - total_demand_minus_supply["STRAWBERRY"],
        "EGG" : obs["market"]["inventory"].get("EGG", 0) - total_demand_minus_supply["EGG"],
        "MILK" : obs["market"]["inventory"].get("MILK", 0) - total_demand_minus_supply["MILK"],
        "WOOL" : obs["market"]["inventory"].get("WOOL", 0) - total_demand_minus_supply["WOOL"]
    }

    projected_prices = {
        "WHEAT" : predict_price(projected_inventory["WHEAT"], 25, 400, "sqrt", 0.80, "log", 0.20),
        "CARROT" : predict_price(projected_inventory["CARROT"], 35, 450, "hinge", 1, "sqrt", 0.70),
        "MELON" : predict_price(projected_inventory["MELON"], 250, 300, "log", 0.20, "sq", 3.6),
        "TOMATO" : predict_price(projected_inventory["TOMATO"], 60, 200, "hinge", 0.40, "sqrt", 0.60),
        "STRAWBERRY" : predict_price(projected_inventory["STRAWBERRY"], 120, 100, "sqrt", 0.70, "linear", 1.60),
        "EGG" : predict_price(projected_inventory["EGG"], 50, 332, "hinge", 0.40, "log", 0.20),
        "MILK" : predict_price(projected_inventory["MILK"], 160, 122, "sqrt", 0.60, "linear", 1.60),
        "WOOL" : predict_price(projected_inventory["WOOL"], 200, 105, "log", 0.20, "sq", 3.20)
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
    product_values = {
        "WHEAT" : projected_prices["WHEAT"] * 6 / 4,
        "CARROT" : projected_prices["CARROT"] * 4 / 3,
        "MELON" : projected_prices["MELON"] * 6 / 10,
        "TOMATO" : projected_prices["TOMATO"] * 16 / 11,
        "STRAWBERRY" : projected_prices["STRAWBERRY"] * 16 / 16,
        "EGG" : projected_prices["EGG"] * (26 - day) / (30 - day) if 26 - day > 0 else 0,
        "MILK" : projected_prices["MILK"] * (22 - day) / (30 - day) if 22 - day > 0 else 0,
        "WOOL" : projected_prices["WOOL"] * (24 - day) / (30 - day) if 24 - day > 0 else 0
    }
            
    # 1. State Extraction & Tracking
    # Parse coordinates, tile statuses, cash, and shop multipliers
    farmer_pos = my_farm["farmer"]
    hands_list = my_farm["hands"]
    worker_actions = []
    market_orders = []
    # 2. Market Execution (Hourly / Daily triggers)
    # Buy seeds, sell ready produce, hire farm hands
    if shed.get("WHEAT", 0) > 0:
        market_orders.append(["SELL", "WHEAT", 1])
    if hour == 0:
        # e.g., buy seeds or hire hands at the start of a new day
        if len(my_farm["unlocked_quadrants"]) == 1:
            for i in range(5):
                market_orders.append(["HIRE"])
        #if day == 0:
            #product_values["MELON"] += 100
            #if my_farm["money"] >= 200 and seeds.get("MELON", 0) < 5:
            #    market_orders.append(["BUY_SEED", "MELON", 5])
        

        #market_orders.append(["BUY_SEED", "WHEAT", 10])
        worker_actions.append(["PASS"]) #why not move

    # 3. Task Selection & Route Execution
    # Determine what the farmer standing on (farmer_pos[0], farmer_pos[1]) needs to do
    elif hour < 24:
        if hour == 1:
            wheat_seed_wanted = 0
            carrot_seed_wanted = 0
            melon_seed_wanted = 0
            tomato_seed_wanted = 0
            strawberry_seed_wanted = 0
            goose_wanted = 0
            cow_wanted = 0
            sheep_wanted = 0
            total_seeds_and_animals = seeds.get("WHEAT", 0) + seeds.get("CARROT", 0) + seeds.get("MELON", 0) + seeds.get("TOMATO", 0) + seeds.get("STRAWBERRY", 0) + shed.get("GOOSE", 0) + shed.get("COW", 0) + shed.get("SHEEP", 0)
            start_money = my_farm["money"]
            while total_seeds_and_animals < empty_tiles and start_money > 0:
                current_max_value_product = max(zip(product_values.values(), product_values.keys()))[1]
                if current_max_value_product == "WHEAT":
                    wheat_seed_wanted += 1
                    start_money -= prices["WHEAT"]
                elif current_max_value_product == "CARROT":
                    carrot_seed_wanted += 1
                    start_money -= prices["CARROT"]
                elif current_max_value_product == "MELON":
                    melon_seed_wanted += 1
                    start_money -= prices["MELON"]
                elif current_max_value_product == "TOMATO":
                    tomato_seed_wanted += 1
                    start_money -= prices["TOMATO"]
                elif current_max_value_product == "STRAWBERRY":
                    strawberry_seed_wanted += 1
                    start_money -= prices["STRAWBERRY"]
                elif current_max_value_product == "EGG":
                    goose_wanted += 1
                    start_money -= prices["EGG"]
                elif current_max_value_product == "MILK":
                    cow_wanted += 1
                    start_money -= prices["MILK"]
                elif current_max_value_product == "WOOL":
                    sheep_wanted += 1
                    start_money -= prices["WOOL"]
                else:
                    print('something went wrong with current_max_value_product:', current_max_value_product)
                    break
                total_seeds_and_animals += 1
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
            if goose_wanted > 0:
                market_orders.append(["BUY_ANIMAL", "GOOSE", goose_wanted])
            if cow_wanted > 0:
                market_orders.append(["BUY_ANIMAL", "COW", cow_wanted])
            if sheep_wanted > 0:
                market_orders.append(["BUY_ANIMAL", "SHEEP", sheep_wanted])
        if len(my_farm["unlocked_quadrants"]) == 1:
            for i in range(len(hands_list) + 1):
                if i == 0:
                    current_x, current_y = farmer_pos
                else:
                    current_x, current_y = hands_list[i - 1]
                current_tile = my_tiles[current_y][current_x]
                if i == 0:
                    if current_x != 0:
                        worker_actions.append(step_toward(farmer_pos, (0, 5)))
                    else:
                        if current_tile == None:
                            if "COW" in inventories[0] or "SHEEP" in inventories[0]:
                                worker_actions.append(["BUILD_PASTURE"])
                            elif "GOOSE" in inventories[0]:
                                worker_actions.append(["BUILD_COOP"])
                            elif seeds.get("WHEAT", 0) > 0:
                                worker_actions.append(["PLANT", "WHEAT"])

                        elif current_tile.get("kind") == "PLANT":
                            if current_tile.get("consecutive_unwatered") > 0 and not current_tile.get("watered_today"):
                                worker_actions.append(["WATER"])
                            elif current_tile.get("crop") == "WHEAT":
                                if day - current_tile.get("planted_day") >= 2: #bonus window of wheat
                                    if not current_tile.get("watered_today"):
                                        worker_actions.append(["WATER"])
                                    elif current_tile.get("fertilized_until_day") == -1 and inventories[0].get("FERTILIZER", 0) > 0:
                                        worker_actions.append(["FERTILIZE"])
                                    elif day - current_tile.get("planted_day") >= 4 or current_tile.get("yield_units") >= 6:
                                        worker_actions.append(["HARVEST"])
                                    else:
                                        if current_y > 0:
                                            worker_actions.append(["NORTH"])
                                        else:
                                            worker_actions.append(["PASS"])
                                else:
                                    if current_y > 0:
                                        worker_actions.append(["NORTH"])
                                    else:
                                        worker_actions.append(["PASS"])
                else:
                    worker_actions.append(["PASS"])
            


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
    return {
        "farmer": worker_actions[0] if worker_actions else "PASS",
        "hands": [] if len(worker_actions) < 2  else worker_actions[1:],
        "market": market_orders,
    }