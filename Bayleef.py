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
    tiles = my_farm["tiles"]
    private = obs["private"]
    inventories = private["inventories"]

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

    product_values = {
            "WHEAT" : daily_demand["WHEAT"],
            "CARROT" : daily_demand["CARROT"],
            "MELON" : daily_demand["MELON"],
            "TOMATO" : daily_demand["TOMATO"],
            "STRAWBERRY" : daily_demand["STRAWBERRY"],
            "EGG" : daily_demand["EGG"],
            "MILK" : daily_demand["MILK"],
            "WOOL" : daily_demand["WOOL"]
        }
    empty_tiles = 0
    for tile_row in tiles:
        for tile in tile_row:
            if tile == None:
                empty_tiles += 1
                continue
            elif tile == "LOCKED":
                continue
            elif tile.get("kind") == "PLANT":
                continue
            elif tile.get("kind") == "WEED":
                continue
            elif tile.get("kind") == "COOP":
                continue
            elif tile.get("kind") == "PASTURE":
                continue
            else:
                print('unexpected tile kind:', tile.get("kind"))
                continue
            
    # 1. State Extraction & Tracking
    # Parse coordinates, tile statuses, cash, and shop multipliers
    farmer_pos = my_farm["farmer"]
    hands_list = my_farm["hands"]
    worker_actions = []
    market_orders = []
    # 2. Market Execution (Hourly / Daily triggers)
    # Buy seeds, sell ready produce, hire farm hands
    if hour == 0:
        # e.g., buy seeds or hire hands at the start of a new day
        if len(my_farm["unlocked_quadrants"]) == 1:
            for i in range(5):
                market_orders.append(["HIRE"])
        if day == 0:
            if my_farm["money"] >= 200 and obs["private"]["seeds"].get("MELON", 0) < 5:
                market_orders.append(["BUY_SEED", "MELON", 5])
            worker_actions.append(["WEST"])

    # 3. Task Selection & Route Execution
    # Determine what the farmer standing on (farmer_pos[0], farmer_pos[1]) needs to do
    elif hour < 24:
        if len(my_farm["unlocked_quadrants"]) == 1:
            for i in range(len(hands_list) + 1):
                if i == 0:
                    current_x, current_y = farmer_pos
                else:
                    current_x, current_y = hands_list[i - 1]
                current_tile = my_farm["tiles"][current_y][current_x]
                if i == 0:
                    if current_x != 0:
                        worker_actions.append(step_toward(farmer_pos, (0, 5)))
                    else:
                        if current_tile == None:
                            print("empty")
                        elif current_tile.get("kind") == "PLANT":
                            if current_tile.get("crop") == "WHEAT":
                                if current_tile.get("planted_day") - day >= 2: #bonus window of wheat
                                    if not current_tile.get("watered_today"):
                                        worker_actions.append(["WATER"])
                                    elif current_tile.get("fertilized_until_day") == -1 and inventories[0]["FERTILIZER"] > 0:
                                        worker_actions.append(["FERTILIZE"])
                                    elif current_tile.get("planted_day") - day >= 4 or current_tile.get("yield_units") >= 6:
                                        worker_actions.append(["HARVEST"])
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