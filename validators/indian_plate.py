"""
validators/indian_plate.py
==========================
Indian vehicle registration plate validation and positional correction.

Covers:
  - Standard format: SS DD [A-Z]{1,3} DDDD  (e.g. TN 09 AB 1234)
  - BH series:       DD BH DDDD [A-Z]{1,2}  (e.g. 22 BH 1234 AB)
  - Loose legacy:    tolerant fallback for hand-painted / older plates

State/UT codes sourced from MoRTH / Parivahan:
  https://parivahan.gov.in/parivahan/
  https://morth.nic.in/

Positional confusion correction:
  - Letter positions: 0→O, 1→I, 2→Z, 5→S, 8→B, 6→G
  - Digit positions:  O→0, Q→0, D→0, I→1, L→1, Z→2, S→5, B→8, G→6, T→7
  - Max 2 corrections per plate, flagged corrected=True
  - Never invents characters to force validity

Pure functions, no side effects, fully unit-tested.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
#  STATE/UT CODES
# ─────────────────────────────────────────────────────────────────────────────
# Source: MoRTH / Parivahan portal (https://parivahan.gov.in)
# Verified codes are from official RTO listing.
# Codes marked [unverified] are reported but not confirmed on Parivahan.

VERIFIED_STATE_CODES = frozenset({
    # States
    "AN",  # Andaman and Nicobar Islands
    "AP",  # Andhra Pradesh
    "AR",  # Arunachal Pradesh
    "AS",  # Assam
    "BR",  # Bihar
    "CG",  # Chhattisgarh
    "CH",  # Chandigarh
    "DD",  # Dadra and Nagar Haveli and Daman and Diu
    "DL",  # Delhi
    "GA",  # Goa
    "GJ",  # Gujarat
    "HP",  # Himachal Pradesh
    "HR",  # Haryana
    "JH",  # Jharkhand
    "JK",  # Jammu and Kashmir
    "KA",  # Karnataka
    "KL",  # Kerala
    "LA",  # Ladakh
    "MH",  # Maharashtra
    "ML",  # Meghalaya
    "MN",  # Manipur
    "MP",  # Madhya Pradesh
    "MZ",  # Mizoram
    "NL",  # Nagaland
    "OD",  # Odisha
    "PB",  # Punjab
    "PY",  # Puducherry
    "RJ",  # Rajasthan
    "SK",  # Sikkim
    "TN",  # Tamil Nadu
    "TR",  # Tripura
    "TS",  # Telangana
    "UK",  # Uttarakhand
    "UP",  # Uttar Pradesh
    "WB",  # West Bengal
})

# Codes that appear in the wild but are not on the current MoRTH listing
UNVERIFIED_STATE_CODES = frozenset({
    "OR",  # Old code for Odisha (now OD)
    "UA",  # Old code for Uttarakhand (now UK)
    "DN",  # Old code for Dadra and Nagar Haveli
})

ALL_STATE_CODES = VERIFIED_STATE_CODES | UNVERIFIED_STATE_CODES


# ─────────────────────────────────────────────────────────────────────────────
#  CONFUSION MAPS
# ─────────────────────────────────────────────────────────────────────────────

# When a position expects a LETTER, these digit→letter swaps are tried
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G",
}

# When a position expects a DIGIT, these letter→digit swaps are tried
LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "T": "7",
}


# ─────────────────────────────────────────────────────────────────────────────
#  RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlateValidation:
    """Result of plate validation and optional correction."""
    raw_text: str              # Original OCR text (uppercased, stripped)
    normalized: str            # After removing spaces, hyphens
    corrected_text: str        # After positional correction (may == normalized)
    plate_valid: bool          # True if matches a known format
    format_type: str           # "standard" | "bh_series" | "legacy" | "invalid"
    state_code: str            # Extracted state code or ""
    state_verified: bool       # True if state code is in VERIFIED list
    corrected: bool            # True if any characters were changed
    n_corrections: int         # Number of character corrections applied
    needs_review: bool         # True if invalid or unresolved
    confidence_note: str       # Human-readable note


# ─────────────────────────────────────────────────────────────────────────────
#  CORE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

# Standard: SS DD L{1,3} DDDD
# Examples: TN09AB1234, DL01C1234, KA51AD9999
STANDARD_RE = re.compile(
    r"^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$"
)

# BH series: DD BH DDDD L{1,2}
# Examples: 22BH1234AB, 20BH5678A
BH_SERIES_RE = re.compile(
    r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$"
)

# Legacy / loose: SS DD with variable suffix
# Catches older formats, hand-painted, partial reads
LEGACY_RE = re.compile(
    r"^([A-Z]{2})(\d{1,2})([A-Z0-9]{1,7})$"
)


# ─────────────────────────────────────────────────────────────────────────────
#  NORMALIZE
# ─────────────────────────────────────────────────────────────────────────────

def normalize_plate(text: str) -> str:
    """Normalize plate text: uppercase, strip whitespace/hyphens/dots."""
    if not text:
        return ""
    result = text.upper().strip()
    # Remove common separators
    result = re.sub(r"[\s\-\.·•|/\\]", "", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  POSITIONAL CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def _correct_char(char: str, expect_digit: bool) -> Tuple[str, bool]:
    """Attempt to correct a single character based on positional expectation.

    Args:
        char: The character to possibly correct.
        expect_digit: True if this position should be a digit.

    Returns:
        (corrected_char, was_changed)
    """
    if expect_digit:
        if char.isdigit():
            return char, False
        replacement = LETTER_TO_DIGIT.get(char)
        if replacement:
            return replacement, True
        return char, False
    else:
        if char.isalpha():
            return char, False
        replacement = DIGIT_TO_LETTER.get(char)
        if replacement:
            return replacement, True
        return char, False


def correct_standard(text: str) -> Tuple[str, int]:
    """Apply positional correction to a standard-format plate.

    Expected: LL DD L{1,3} DDDD
    Position: 0,1=letter  2,3=digit  4..(n-5)=letter  (n-4)..n=digit

    Returns:
        (corrected_text, n_corrections)
    """
    if len(text) < 9 or len(text) > 12:
        return text, 0

    chars = list(text)
    corrections = 0

    # Positions 0,1: state code (letters)
    for i in [0, 1]:
        c, changed = _correct_char(chars[i], expect_digit=False)
        if changed:
            chars[i] = c
            corrections += 1

    # Positions 2,3: district code (digits)
    for i in [2, 3]:
        c, changed = _correct_char(chars[i], expect_digit=True)
        if changed:
            chars[i] = c
            corrections += 1

    # Last 4: registration number (digits)
    for i in range(len(chars) - 4, len(chars)):
        c, changed = _correct_char(chars[i], expect_digit=True)
        if changed:
            chars[i] = c
            corrections += 1

    # Middle: series code (letters)
    for i in range(4, len(chars) - 4):
        c, changed = _correct_char(chars[i], expect_digit=False)
        if changed:
            chars[i] = c
            corrections += 1

    return "".join(chars), corrections


def correct_bh_series(text: str) -> Tuple[str, int]:
    """Apply positional correction to a BH-series plate.

    Expected: DD BH DDDD L{1,2}

    Returns:
        (corrected_text, n_corrections)
    """
    if len(text) < 9 or len(text) > 10:
        return text, 0

    chars = list(text)
    corrections = 0

    # Positions 0,1: year (digits)
    for i in [0, 1]:
        c, changed = _correct_char(chars[i], expect_digit=True)
        if changed:
            chars[i] = c
            corrections += 1

    # Positions 2,3: "BH" (letters)
    for i in [2, 3]:
        c, changed = _correct_char(chars[i], expect_digit=False)
        if changed:
            chars[i] = c
            corrections += 1

    # Positions 4-7: number (digits)
    for i in range(4, 8):
        c, changed = _correct_char(chars[i], expect_digit=True)
        if changed:
            chars[i] = c
            corrections += 1

    # Positions 8+: class (letters)
    for i in range(8, len(chars)):
        c, changed = _correct_char(chars[i], expect_digit=False)
        if changed:
            chars[i] = c
            corrections += 1

    return "".join(chars), corrections


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def validate_plate(raw_text: str) -> PlateValidation:
    """Validate and optionally correct an Indian licence plate string.

    Strategy:
      1. Normalize (uppercase, strip separators)
      2. Try strict match (standard, BH only — NOT legacy)
      3. If no strict match, apply positional correction (max 2 edits)
      4. Re-try strict match on corrected text
      5. Try legacy match as last resort (for hand-painted / old plates)
      6. If still no match, return plate_valid=False, needs_review=True

    Order matters: correction before legacy, because inputs like TN09AB12O4
    should be corrected to TN09AB1204 (standard), not accepted as legacy.

    Never invents characters to force validity.

    Args:
        raw_text: Raw OCR output text.

    Returns:
        PlateValidation with all fields populated.
    """
    normalized = normalize_plate(raw_text)

    if not normalized:
        return PlateValidation(
            raw_text=raw_text,
            normalized="",
            corrected_text="",
            plate_valid=False,
            format_type="invalid",
            state_code="",
            state_verified=False,
            corrected=False,
            n_corrections=0,
            needs_review=True,
            confidence_note="Empty plate text",
        )

    # ── 1. Try strict match (standard + BH only) ────────────────────────
    result = _try_match_strict(normalized, corrected=False, n_corrections=0, raw_text=raw_text)
    if result.plate_valid:
        return result

    # ── 2. Try positional correction → strict match ─────────────────────

    # Try standard correction
    corrected_std, n_std = correct_standard(normalized)
    if n_std > 0 and n_std <= 2:
        result_std = _try_match_strict(corrected_std, corrected=True, n_corrections=n_std, raw_text=raw_text)
        if result_std.plate_valid:
            return result_std

    # Try BH correction
    corrected_bh, n_bh = correct_bh_series(normalized)
    if n_bh > 0 and n_bh <= 2:
        result_bh = _try_match_strict(corrected_bh, corrected=True, n_corrections=n_bh, raw_text=raw_text)
        if result_bh.plate_valid:
            return result_bh

    # ── 3. Legacy fallback (uncorrected only) ────────────────────────────
    result_legacy = _try_match_legacy(normalized, raw_text=raw_text)
    if result_legacy.plate_valid:
        return result_legacy

    # ── 4. Unresolved ────────────────────────────────────────────────────
    return PlateValidation(
        raw_text=raw_text,
        normalized=normalized,
        corrected_text=normalized,
        plate_valid=False,
        format_type="invalid",
        state_code=normalized[:2] if len(normalized) >= 2 and normalized[:2].isalpha() else "",
        state_verified=False,
        corrected=False,
        n_corrections=0,
        needs_review=True,
        confidence_note="No valid format matched after correction attempts",
    )


def _try_match_strict(
    text: str,
    corrected: bool,
    n_corrections: int,
    raw_text: str,
) -> PlateValidation:
    """Try to match text against strict formats (standard + BH only).

    Does NOT try legacy — that's a separate fallback after correction.

    Returns:
        PlateValidation with plate_valid=True if matched, False otherwise.
    """
    # Standard: SS DD L{1,3} DDDD
    m = STANDARD_RE.match(text)
    if m:
        state = m.group(1)
        state_verified = state in VERIFIED_STATE_CODES
        state_known = state in ALL_STATE_CODES
        return PlateValidation(
            raw_text=raw_text,
            normalized=normalize_plate(raw_text),
            corrected_text=text,
            plate_valid=True,
            format_type="standard",
            state_code=state,
            state_verified=state_verified,
            corrected=corrected,
            n_corrections=n_corrections,
            needs_review=not state_known,
            confidence_note=_state_note(state, state_verified, state_known),
        )

    # BH series: DD BH DDDD L{1,2}
    m = BH_SERIES_RE.match(text)
    if m:
        return PlateValidation(
            raw_text=raw_text,
            normalized=normalize_plate(raw_text),
            corrected_text=text,
            plate_valid=True,
            format_type="bh_series",
            state_code="BH",
            state_verified=True,  # BH is a national series
            corrected=corrected,
            n_corrections=n_corrections,
            needs_review=False,
            confidence_note="BH (Bharat) series — valid national format",
        )

    # No strict match
    return PlateValidation(
        raw_text=raw_text,
        normalized=normalize_plate(raw_text),
        corrected_text=text,
        plate_valid=False,
        format_type="invalid",
        state_code="",
        state_verified=False,
        corrected=False,
        n_corrections=0,
        needs_review=True,
        confidence_note="",
    )


def _try_match_legacy(text: str, raw_text: str) -> PlateValidation:
    """Try to match text against legacy/loose format (last resort).

    Only used after strict matching and correction have been tried.
    Legacy matches always set needs_review=True.
    """
    m = LEGACY_RE.match(text)
    if m:
        state = m.group(1)
        state_known = state in ALL_STATE_CODES
        if state_known and len(text) >= 6:
            return PlateValidation(
                raw_text=raw_text,
                normalized=normalize_plate(raw_text),
                corrected_text=text,
                plate_valid=True,
                format_type="legacy",
                state_code=state,
                state_verified=state in VERIFIED_STATE_CODES,
                corrected=False,
                n_corrections=0,
                needs_review=True,  # Legacy always needs review
                confidence_note=f"Legacy/non-standard format from {state}",
            )

    return PlateValidation(
        raw_text=raw_text,
        normalized=normalize_plate(raw_text),
        corrected_text=text,
        plate_valid=False,
        format_type="invalid",
        state_code="",
        state_verified=False,
        corrected=False,
        n_corrections=0,
        needs_review=True,
        confidence_note="",
    )


def _state_note(state: str, verified: bool, known: bool) -> str:
    """Generate a note about the state code."""
    if verified:
        return f"Verified state code: {state}"
    if known:
        return f"Known but unverified code: {state} (old/alternate)"
    return f"Unknown state code: {state} — needs review"
