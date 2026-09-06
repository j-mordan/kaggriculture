"""
Kaggriculture agent.

Schema (confirmed against a live env.reset()):
  observation = {
    "step": int, "player": 0|1, "day": int, "hour": int,
    "farms": [farm0, farm1],
    "market": {"inventory": {...}, "prices": {...}},
    "town": {"unlocked_shops": [...]},
    "private": {"shed": {...}, "seeds": {...}, "inventories": [...]},
  }
  farm = {
    "money": float, "tiles": [[tile,...]...] (tiles[y][x]),
    "farmer": [x, y], "hands": [[x, y], ...],
    "unlocked_quadrants": [...], "hires_today": int,
  }
  action returned by agent = {"farmer": [...], "hands": [[...], ...], "market": [...]}
"""

CROPS = {
    # crop: (seed_cost, base_price, first_yield_day, revenue_per_tile_per_day)
    "WHEAT":      dict(seed_cost=10,  price=25,  first_yield=2,  rev=20.0),
    "CARROT":     dict(seed_cost=20,  price=35,  first_yield=2,  rev=26.25),
    "TOMATO":     dict(seed_cost=50,  price=60,  first_yield=8,  rev=19.8),
    "STRAWBERRY": dict(seed_cost=100, price=120, first_yield=10, rev=28.8),
    "MELON":      dict(seed_cost=80,  price=250, first_yield=10, rev=137.5),
}

FAST_CASH_CROPS = {"WHEAT", "CARROT"}  # first_yield <= 2 days
LOW_CASH_THRESHOLD = 400  # below this, only plant fast-turnaround crops

# Order in which we prefer to plant, biased toward good $/tile/day but
# capped so we don't dump too much into one crop and crash its price.
CROP_ORDER = ["WHEAT", "CARROT", "MELON", "STRAWBERRY", "TOMATO"]

QUADRANT_ORIGIN = {  # top-left corner (x, y) of each 5x5 quadrant on a 10x10 board
    "NW": (0, 0), "NE": (5, 0), "SW": (0, 5), "SE": (5, 5),
}

MIN_CASH_RESERVE = 50
HIRE_CASH_FLOOR = 250        # hire early -- more units = more tended tiles = more income
LAND_CASH_FLOOR = 1200       # only expand once we have hands to actually tend the new land
MIN_HANDS_FOR_LAND = 2       # don't buy land until we can staff it
SEED_STOCK_TARGET = 1        # buy seeds one at a time, only right before planting
MAX_TILES_PER_UNIT = 3       # cap concurrently-growing tiles so nothing gets left unwatered
CROP_SWITCH_MARGIN = 1.25    # only switch active crop if it's >25% better (avoid thrashing)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def step_toward(pos, target, board_size):
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


def find_tiles(tiles, predicate):
    """Yield (x, y) for tiles matching predicate(tile)."""
    out = []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if predicate(t):
                out.append((x, y))
    return out


def nearest(pos, candidates):
    if not candidates:
        return None
    px, py = pos
    return min(candidates, key=lambda c: abs(c[0] - px) + abs(c[1] - py))


def crop_scores(prices):
    scores = {}
    for crop in CROP_ORDER:
        info = CROPS[crop]
        cur_price = prices.get(crop, info["price"])
        scores[crop] = info["rev"] * (cur_price / info["price"])
    return scores


def dominant_planted_crop(tiles):
    """What crop do we already have the most of growing? Used to keep the
    farm consistent instead of flip-flopping crop choice every turn."""
    counts = {}
    for row in tiles:
        for t in row:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                c = t.get("crop")
                counts[c] = counts.get(c, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def choose_crop(tiles, prices, money):
    """Pick a crop and stick with it (hysteresis) unless another crop is
    meaningfully better right now. This avoids buying seeds for five
    different crops and never finishing any of them.

    While cash is low, restrict to fast-turnaround crops (wheat/carrot,
    first yield in 2 days) so we don't lock money into a 10-day melon
    with nothing coming in the meantime.
    """
    scores = crop_scores(prices)
    if money < LOW_CASH_THRESHOLD:
        scores = {c: s for c, s in scores.items() if c in FAST_CASH_CROPS}
    current = dominant_planted_crop(tiles)
    best_crop = max(scores, key=scores.get)
    if current is None or current not in scores:
        return best_crop
    if scores.get(best_crop, 0) > scores.get(current, 0) * CROP_SWITCH_MARGIN:
        return best_crop
    return current


def is_harvest_ready(tile, day):
    """`yield_units` on a plant tile can be populated before the crop is
    actually old enough to harvest (it looks like a projected/potential
    yield field, not strictly 'harvestable right now'). Gate on crop age
    vs. that crop's first_yield_day too, or HARVEST silently no-ops while
    the plant sits unwatered and dies."""
    if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
        return False
    if tile.get("yield_units", 0) <= 0:
        return False
    crop = tile.get("crop")
    info = CROPS.get(crop)
    if info is None:
        return True  # unknown crop type, don't block on age
    age = day - tile.get("planted_day", day)
    return age >= info["first_yield"]


def decide_unit_action(pos, tiles, board_size, seeds, active_crop, targets, allow_plant, day):
    """Decide what a single unit (farmer or hand) standing at `pos` should do.
    `targets` is a dict of candidate tile lists: harvestable, needs_water, empty, weeds.
    `allow_plant` gates new planting once we're at our tending-capacity cap,
    so a lone unit doesn't scatter plants it can't keep watered.
    Returns an action list, e.g. ["HARVEST"], ["WATER"], ["PLANT","WHEAT"], or a move.
    """
    x, y = pos
    tile = tiles[y][x]

    # 1. Standing on something actionable right now. Watering/harvesting
    #    existing plants always takes priority over starting new ones.
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if is_harvest_ready(tile, day):
            return ["HARVEST"]
        if not tile.get("watered_today", False):
            return ["WATER"]
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]
    if tile is None and allow_plant:
        if seeds.get(active_crop, 0) > 0:
            return ["PLANT", active_crop]

    # 2. Otherwise, walk toward the closest interesting tile. Watering
    #    existing plants beats chasing new empty tiles.
    order = ("harvestable", "needs_water", "weeds")
    if allow_plant:
        order = order + ("empty",)
    for key in order:
        dest = nearest(pos, targets[key])
        if dest is not None and dest != pos:
            mv = step_toward(pos, dest, board_size)
            if mv:
                return mv

    return ["PASS"]


def agent(obs):
    player = obs["player"]
    farm = obs["farms"][player]
    private = obs["private"]
    tiles = farm["tiles"]
    board_size = len(tiles)
    money = farm["money"]
    prices = obs["market"]["prices"]
    market_inventory = obs["market"]["inventory"]
    shed = private["shed"]
    seeds = private["seeds"]

    day = obs.get("day", 0)
    active_crop = choose_crop(tiles, prices, money)

    # --- Build target tile lists once per turn ---
    def is_unlocked_plant(t):
        return isinstance(t, dict) and t.get("kind") == "PLANT"

    harvestable = find_tiles(tiles, lambda t: is_harvest_ready(t, day))
    needs_water = find_tiles(tiles, lambda t: is_unlocked_plant(t) and not t.get("watered_today", False))
    weeds = find_tiles(tiles, lambda t: isinstance(t, dict) and t.get("kind") == "WEED")
    empty = find_tiles(tiles, lambda t: t is None)
    growing_count = len(harvestable) + len(needs_water)  # plants currently alive
    targets = {
        "harvestable": harvestable,
        "needs_water": needs_water,
        "weeds": weeds,
        "empty": empty,
    }

    num_units = 1 + len(farm["hands"])
    tending_cap = num_units * MAX_TILES_PER_UNIT
    allow_plant = growing_count < tending_cap

    # --- Farmer action ---
    farmer_pos = tuple(farm["farmer"])
    farmer_action = decide_unit_action(farmer_pos, tiles, board_size, seeds, active_crop, targets, allow_plant, day)

    # --- Hand actions (each hand acts independently; avoid piling everyone
    #     onto the exact same target by removing it from the pool once used
    #     this turn, cheaply) ---
    hand_actions = []
    used_targets = set()
    for hpos in farm["hands"]:
        hpos = tuple(hpos)
        local_targets = {
            k: [t for t in v if t not in used_targets] for k, v in targets.items()
        }
        act = decide_unit_action(hpos, tiles, board_size, seeds, active_crop, local_targets, allow_plant, day)
        hand_actions.append(act)
        if act and act[0] in ("HARVEST", "WATER", "PLANT", "DIG"):
            used_targets.add(hpos)

    # --- Market orders ---
    market_orders = []

    # Sell everything sitting in the shed.
    for good, qty in shed.items():
        if good == "FERTILIZER":
            continue
        if qty > 0:
            market_orders.append(["SELL", good, qty])

    # Buy seed for the active crop only when we actually have room to plant
    # it (empty tile available, under our tending cap) and don't already
    # hold one. One at a time avoids stockpiling unused seed for crops we
    # then abandon.
    if (
        allow_plant
        and empty
        and money > MIN_CASH_RESERVE
        and seeds.get(active_crop, 0) < SEED_STOCK_TARGET
    ):
        cost = CROPS[active_crop]["seed_cost"]
        if money - cost > MIN_CASH_RESERVE:
            market_orders.append(["BUY_SEED", active_crop, 1])

    # Hire another hand once we're comfortably capitalized -- more units
    # tending the same land is one of the biggest income multipliers.
    hires_today = farm.get("hires_today", 0)
    if money > HIRE_CASH_FLOOR and hires_today < 4:
        market_orders.append(["HIRE"])

    # Expand land only once we have enough hands to actually work it --
    # otherwise new tiles just grow weeds. Try quadrants in a fixed order.
    unlocked = set(farm.get("unlocked_quadrants", []))
    if money > LAND_CASH_FLOOR and num_units >= MIN_HANDS_FOR_LAND:
        for quad in ("NE", "SW", "SE"):
            if quad not in unlocked:
                market_orders.append(["BUY_LAND", quad])
                break

    market_orders = market_orders[:10]  # respect maxMarketOrdersPerTurn

    return {
        "farmer": farmer_action,
        "hands": hand_actions,
        "market": market_orders,
    }