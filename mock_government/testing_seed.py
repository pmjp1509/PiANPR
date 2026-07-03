"""
mock_government/testing_seed.py
===============================
TESTING ONLY.  Auto-populates the mock challan DB with random data so we can
demo without real government records. This is NOT part of the production
service -- it deliberately lives in its own file with a single public function.

Rule for a plate the DB has never seen:
    70% chance -> amount = 0        (no challan)
    30% chance -> amount = random 500..5000
Plates already in the DB are never touched.

This file is invoked ONLY by challan_service.py, and only while
ENABLE_TESTING is True there. main.py never imports it.

>>> FOR PRODUCTION <<<
Set ENABLE_TESTING = False in challan_service.py, then delete this file.
Because challan_service imports it lazily (inside the ENABLE_TESTING guard),
deleting this file cannot break anything once the flag is False.
"""

import random

from . import database


def seed_if_new(plate):
    """If `plate` is not already in the mock DB, insert it with a random amount."""
    if not plate or database.exists(plate):
        return
    amount = random.randint(500, 5000) if random.random() < 0.30 else 0
    database.insert(plate, amount)
