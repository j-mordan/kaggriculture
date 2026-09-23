def agent(obs):
    worker_actions = []
    market_orders = []
    if obs.get("hour") == 0:
        for i in range(3):
            market_orders.append(["HIRE"])
    return {
            "farmer": worker_actions[0] if worker_actions else "PASS",
            "hands": [] if len(worker_actions) < 2  else worker_actions[1:],
            "market": market_orders,
        }