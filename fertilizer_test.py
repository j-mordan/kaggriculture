def agent(obs):
    worker_actions = []
    market_orders = []
    p = obs.get("player")
    f = obs.get("farms")[p]
    if obs.get("hour") == 0:
        worker_actions.append(["PASS"])
        market_orders.append(["BUY_SEED", "WHEAT", 1])
        market_orders.append(["BUY_PRODUCT", "FERTILIZER", 1])
    elif obs.get("hour") == 1:
        worker_actions.append(["PICKUP", "FERTILIZER", 1])
    elif obs.get("hour") == 2:
        worker_actions.append(["PLANT", "WHEAT"])
    elif obs.get("hour") == 3:
        worker_actions.append(["FERTILIZE"])
        print(f.get("tiles")[4][4].get("fertilized_until_day"))
    elif obs.get("hour") == 4:
        print(f.get("tiles")[4][4].get("fertilized_until_day"))

    return {
            "farmer": worker_actions[0] if worker_actions else "PASS",
            "hands": [] if len(worker_actions) < 2  else worker_actions[1:],
            "market": market_orders,
        }