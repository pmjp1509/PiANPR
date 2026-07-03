"""
mock_government/challan_service.py
==================================
THE ONLY entry point the ANPR pipeline is allowed to use:

    from mock_government.challan_service import check_challan
    result = check_challan(plate)
    #   -> {"plate": "TN11A6701", "has_challan": True, "amount": 1500}

The pipeline does NOT know whether the data came from SQLite, the mock seeder,
or a real government server. Everything (including test seeding) is hidden
behind this one function.

>>> PRODUCTION SWAP <<<
1) Set ENABLE_TESTING = False  (stops all mock seeding).
2) Replace the body of check_challan() below with your real HTTP call, e.g.:

       import requests
       def check_challan(plate):
           resp = requests.get(f"https://govapi.example/challans/{plate}", timeout=5)
           resp.raise_for_status()
           data = resp.json()
           amount = int(data.get("total_amount", 0))
           return {"plate": plate, "has_challan": amount > 0, "amount": amount}

3) You may then delete database.py, testing_seed.py and mock_challan.db.

The ANPR pipeline stays exactly the same and imports nothing new.
"""

from . import database

# ---------------------------------------------------------------------------
# TESTING SWITCH
#   True  -> unseen plates are auto-populated with random mock data (demo).
#   False -> behaves like the real service: no seeding. Set False for
#            production, after which testing_seed.py can be deleted safely.
# ---------------------------------------------------------------------------
ENABLE_TESTING = True


def check_challan(plate):
    """
    Look up pending challans for a plate.

    Returns an API-shaped dict (like a real government response):
        {"plate": <str>, "has_challan": <bool>, "amount": <int rupees>}
    has_challan == False (amount 0) means no challan on record / plate not found.
    """
    if ENABLE_TESTING:
        # Local import so production (ENABLE_TESTING=False) never needs this file.
        from .testing_seed import seed_if_new
        seed_if_new(plate)

    amount = database.get_amount(plate)
    amount = amount if amount is not None else 0
    return {"plate": plate, "has_challan": amount > 0, "amount": amount}
