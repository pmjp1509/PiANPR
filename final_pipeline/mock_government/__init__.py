"""
mock_government
===============
A SELF-CONTAINED mock of a government challan API, for demonstration only.

The ANPR pipeline talks to this package through ONE function:

    from mock_government.challan_service import check_challan
    amount = check_challan(plate)      # int rupees, 0 = no challan / not found

In production this whole folder is replaced by a real REST API call. See the
"PRODUCTION SWAP" notes in challan_service.py. Nothing in the ANPR pipeline
knows or cares whether the data came from SQLite or a government server.
"""
