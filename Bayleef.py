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

    demand_EV_Day = { #every time a shop opens, demand spikes by this much per day, fix this
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

    demand_minus_supply = {
        "WHEAT" : daily_demand["WHEAT"] - projected_produced["WHEAT"],
        "CARROT" : daily_demand["CARROT"] - projected_produced["CARROT"],
        "MELON" : daily_demand["MELON"] - projected_produced["MELON"],
        "TOMATO" : daily_demand["TOMATO"] - projected_produced["TOMATO"],
        "STRAWBERRY" : daily_demand["STRAWBERRY"] - projected_produced["STRAWBERRY"],
        "EGG" : daily_demand["EGG"] - projected_produced["EGG"],
        "MILK" : daily_demand["MILK"] - projected_produced["MILK"],
        "WOOL" : daily_demand["WOOL"] - projected_produced["WOOL"]
    }

    projected_prices = {
        "WHEAT" : prices["WHEAT"],
    }

    product_values = { #subtract value based on how many of that crop we have, later
            "WHEAT" : ((daily_demand["WHEAT"] + demand_EV_Day["WHEAT"] * (30 - day)) * prices["WHEAT"]) * 6 / 4,
            "CARROT" : ((daily_demand["CARROT"] + demand_EV_Day["CARROT"] * (30 - day)) * prices["CARROT"]) * 4 / 3,
            "MELON" : ((daily_demand["MELON"]  + demand_EV_Day["MELON"] * (30 - day))  * prices["MELON"]) * 6 / 10,
            "TOMATO" : ((daily_demand["TOMATO"] + demand_EV_Day["TOMATO"] * (30 - day)) * prices["TOMATO"]) * 8 / 11,
            "STRAWBERRY" : ((daily_demand["STRAWBERRY"] + demand_EV_Day["STRAWBERRY"] * (30 - day))  * prices["STRAWBERRY"]) * 8 / 16,
            "EGG" : ((daily_demand["EGG"] + demand_EV_Day["EGG"] * (30 - day)) * prices["EGG"]) * (26 - day) / ((30 - day) if 30 - day > 0 else 1),
            "MILK" : ((daily_demand["MILK"] + demand_EV_Day["MILK"] * (30 - day)) * prices["MILK"]) * (22 - day) / ((30 - day) if 30 - day > 0 else 1),
            "WOOL" : ((daily_demand["WOOL"] + demand_EV_Day["WOOL"] * (30 - day)) * prices["WOOL"]) * (24 - day) / ((30 - day) if 30 - day > 0 else 1)
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
        if day == 0:
            product_values["MELON"] += 100
            #if my_farm["money"] >= 200 and seeds.get("MELON", 0) < 5:
            #    market_orders.append(["BUY_SEED", "MELON", 5])
        wheat_seed_wanted = 0
        carrot_seed_wanted = 0
        melon_seed_wanted = 0
        tomato_seed_wanted = 0
        strawberry_seed_wanted = 0
        goose_wanted = 0
        cow_wanted = 0
        sheep_wanted = 0
        market_orders.append(["BUY_SEED", "WHEAT", 10])
        while empty_tiles > 0 and my_farm["money"] > 0:
            break
        worker_actions.append(["PASS"])
        

    # 3. Task Selection & Route Execution
    # Determine what the farmer standing on (farmer_pos[0], farmer_pos[1]) needs to do
    elif hour < 24:
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