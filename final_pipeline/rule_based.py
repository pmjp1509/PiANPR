"""
rule_based.py
=============
Rule-based correction of OCR output for Indian number plates.

OCR models confuse look-alike characters (6<->G, H<->M, 0<->O, 1<->I, 5<->S,
8<->B ...). A single regex either accepts or rejects -- it cannot *repair*.
This module instead:

  1. Normalises the raw string (upper, strip non-alphanumeric).
  2. Tries each plausible plate FORMAT (standard / BH-series) given the length.
  3. Applies POSITION-AWARE confusion correction:
        - where a LETTER is expected, map stray digits -> letters
        - where a DIGIT  is expected, map stray letters -> digits
  4. Corrects the 2-letter STATE code to the nearest valid state/UT
     (confusion-aware substitution, then edit distance 1).
  5. Optionally validates the 2-digit DISTRICT (RTO) code per state.
  6. Scores every candidate by total correction "penalty" and returns the
     valid candidate needing the fewest, most-likely fixes.

Indian plate grammar
  Standard : SS DD L[1..3] NNNN     e.g. TN 10 AB 1234   (len 9..11)
  BH-series: YY BH NNNN L[1..2]     e.g. 22 BH 1234 AB   (len 9..10)
      SS = state code, DD = district, L = letter series, N = number,
      YY = registration year.

Public API
  correct_plate(raw)  -> dict(plate, valid, score, penalty, corrections, raw, fmt)
  similar(a, b)       -> float ratio in [0,1]
"""

import re
from difflib import SequenceMatcher

import settings as cfg

# ------------------------------------------------------------------
#  State / Union Territory codes (first two letters of a plate)
# ------------------------------------------------------------------
STATE_CODES = {
    "AN": "Andaman and Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "DN": "Dadra and Nagar Haveli",
    "DD": "Daman and Diu", "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat",
    "HR": "Haryana", "HP": "Himachal Pradesh", "JK": "Jammu and Kashmir",
    "KA": "Karnataka", "KL": "Kerala", "LD": "Lakshadweep", "MP": "Madhya Pradesh",
    "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya", "MZ": "Mizoram",
    "NL": "Nagaland", "OR": "Orissa", "PY": "Pondicherry", "PB": "Punjab",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura",
    "UP": "Uttar Pradesh", "WB": "West Bengal",
    # Newer / reorganised states & UTs (kept for completeness)
    "CG": "Chhattisgarh", "JH": "Jharkhand", "UK": "Uttarakhand", "UA": "Uttarakhand",
    "TS": "Telangana", "TG": "Telangana", "LA": "Ladakh",
}
VALID_STATES = set(STATE_CODES.keys())

# ------------------------------------------------------------------
#  District (RTO) code ranges per state -- PARTIAL.
#  Source: en.wikipedia.org/wiki/List_of_Regional_Transport_Office_districts_in_India
#  We store the highest known RTO number; codes above it (for that state) are
#  treated as suspicious and lightly penalised, never rejected. Add states as
#  you verify them. States NOT listed accept any 01..99.
# ------------------------------------------------------------------
RTO_MAX = {
    "TN": 99, "KA": 71, "MH": 55, "DL": 13, "AP": 39, "TS": 38, "KL": 99,
    "GJ": 39, "UP": 96, "RJ": 58, "MP": 70, "HR": 99, "PB": 99, "BR": 56,
    "WB": 99, "OR": 35, "GA": 12, "CH": 4, "JK": 22, "AS": 27, "HP": 99,
}

# ------------------------------------------------------------------
#  Confusion maps
# ------------------------------------------------------------------
# A digit that should have been a letter.
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S",
    "6": "G", "7": "T", "8": "B", "9": "G",
}
# A letter that should have been a digit.
LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "T": "1", "J": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6",
    "B": "8", "C": "0", "E": "8",
}
# Letter <-> letter look-alikes (for fixing the state code).
LETTER_CONFUSION = {
    "O": "QD0U", "Q": "O0", "D": "O0", "U": "OV",
    "I": "L1TJ", "L": "I1", "T": "I1Y", "J": "I1",
    "M": "NH", "N": "MH", "H": "MN",
    "B": "R8E", "R": "BP", "P": "RB",
    "S": "5", "G": "6C", "C": "GO", "Z": "2",
    "V": "UY", "Y": "VT", "K": "X", "X": "K",
    "E": "FB", "F": "EP", "A": "4",
}

LARGE = 100  # penalty for an impossible (non-mappable) conversion


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _force_letter(ch):
    """Return (letter, penalty)."""
    if ch.isalpha():
        return ch, 0
    if ch in DIGIT_TO_LETTER:
        return DIGIT_TO_LETTER[ch], 1
    return ch, LARGE


def _force_digit(ch):
    """Return (digit, penalty)."""
    if ch.isdigit():
        return ch, 0
    if ch in LETTER_TO_DIGIT:
        return LETTER_TO_DIGIT[ch], 1
    return ch, LARGE


def _nearest_state(code):
    """Map a 2-letter code to the nearest valid state. Returns (code, penalty)."""
    if code in VALID_STATES:
        return code, 0
    best, best_pen = code, LARGE
    for vs in VALID_STATES:
        pen = 0
        for a, b in zip(code, vs):
            if a == b:
                continue
            pen += 1 if b in LETTER_CONFUSION.get(a, "") else 2
        if pen < best_pen:
            best, best_pen = vs, pen
    # Accept only a close match (confusion-distance <= 2). +2 so a corrected
    # state always costs more than a clean one.
    if best_pen <= 2:
        return best, best_pen + 2
    return code, LARGE


def _district_penalty(state, dd):
    """Light penalty if the RTO number is suspicious. Never raises on bad input."""
    if not cfg.VALIDATE_DISTRICT:
        return 0
    # District must be exactly two digits; a non-mappable letter can survive
    # _build (e.g. 'N2'), so guard before int().
    if len(dd) != 2 or not dd.isdigit():
        return 2
    if dd == "00":
        return 1
    mx = RTO_MAX.get(state)
    if mx is not None and int(dd) > mx:
        return 1
    return 0


def _build(template, s):
    """
    Apply a position template to string s of equal length.
    template chars: 'L' letter, 'D' digit, literal char = must equal (with fix).
    Returns (corrected_str, penalty, impossible_bool).
    """
    out, pen, impossible = [], 0, False
    for tmpl, ch in zip(template, s):
        if tmpl == "L":
            c, p = _force_letter(ch)
        elif tmpl == "D":
            c, p = _force_digit(ch)
        else:  # literal expected (e.g. 'B','H' in BH series)
            if ch == tmpl:
                c, p = tmpl, 0
            elif tmpl in LETTER_CONFUSION.get(ch, "") or ch in LETTER_CONFUSION.get(tmpl, ""):
                c, p = tmpl, 1
            else:
                c, p = tmpl, LARGE
        if p >= LARGE:
            impossible = True
        out.append(c)
        pen += min(p, LARGE)
    return "".join(out), pen, impossible


def _candidate_standard(s):
    """Standard format SS DD L[1..3] NNNN. len 9..11."""
    n = len(s)
    if n not in (9, 10, 11):
        return None
    series = n - 8                       # 1..3 letters
    template = "LL" + "DD" + "L" * series + "DDDD"
    body, pen, impossible = _build(template, s)
    state, st_pen = _nearest_state(body[:2])
    plate = state + body[2:]
    pen += st_pen
    if st_pen >= LARGE:
        impossible = True
    pen += _district_penalty(state, plate[2:4])
    valid = (not impossible) and (state in VALID_STATES)
    return {"plate": plate, "penalty": pen, "valid": valid, "fmt": "standard"}


def _candidate_bh(s):
    """BH-series YY BH NNNN L[1..2]. len 9..10."""
    n = len(s)
    if n not in (9, 10):
        return None
    series = n - 8                       # 1..2 letters
    template = "DD" + "BH" + "DDDD" + "L" * series
    body, pen, impossible = _build(template, s)
    valid = not impossible
    return {"plate": body, "penalty": pen, "valid": valid, "fmt": "bh"}


def correct_plate(raw):
    """
    Correct a raw OCR string into the most plausible valid Indian plate.

    Returns dict:
        plate       : corrected string
        valid       : bool (format + state satisfied with only plausible fixes)
        score       : confidence in [0,1] (1 = no corrections needed)
        penalty     : total correction cost
        corrections : number of characters changed vs the raw input
        raw         : normalised input
        fmt         : 'standard' | 'bh' | None
    """
    norm = re.sub(r"[^A-Z0-9]", "", str(raw).upper())
    if len(norm) < cfg.MIN_PLATE_LEN:
        return {"plate": norm, "valid": False, "score": 0.0,
                "penalty": LARGE, "corrections": 0, "raw": norm, "fmt": None}

    candidates = []
    for fn in (_candidate_standard, _candidate_bh):
        try:
            c = fn(norm)
        except Exception:
            c = None          # a malformed OCR string must never crash the pipeline
        if c is not None:
            candidates.append(c)
    if not candidates:
        return {"plate": norm, "valid": False, "score": 0.0,
                "penalty": LARGE, "corrections": 0, "raw": norm, "fmt": None}

    # Prefer valid candidates; among those (or all), the lowest penalty wins.
    candidates.sort(key=lambda c: (not c["valid"], c["penalty"]))
    best = candidates[0]

    corrections = sum(1 for a, b in zip(norm, best["plate"]) if a != b)
    score = round(max(0.0, 1.0 - 0.08 * best["penalty"]), 2) if best["valid"] else 0.0
    return {
        "plate": best["plate"], "valid": best["valid"], "score": score,
        "penalty": best["penalty"], "corrections": corrections,
        "raw": norm, "fmt": best["fmt"],
    }


# Quick manual check:  python rule_based.py
if __name__ == "__main__":
    for t in ["TN1OABI234", "TM10AB1234", "KA O5 MG 1234", "22BH1234AB",
              "HR26Dq5551", "6J05AB1234", "tn 10 ab 1234", "ABCDEF"]:
        r = correct_plate(t)
        flag = "OK " if r["valid"] else "BAD"
        print(f"{t:<14} -> {r['plate']:<11} [{flag}] "
              f"fmt={r['fmt']} score={r['score']} fixes={r['corrections']}")
