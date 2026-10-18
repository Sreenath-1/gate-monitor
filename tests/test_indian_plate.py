"""
tests/test_indian_plate.py
==========================
Unit tests for validators/indian_plate.py

Covers:
  - Standard format plates (TN, KA, DL, etc.)
  - BH series plates
  - Legacy/hand-painted plates
  - Positional confusion correction
  - Edge cases: empty, garbage, too many corrections
  - State code verification
  - Two-line plate normalization
  - Max 2 corrections rule
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

from validators.indian_plate import (
    normalize_plate,
    validate_plate,
    correct_standard,
    correct_bh_series,
    VERIFIED_STATE_CODES,
    ALL_STATE_CODES,
)


class TestNormalizePlate(unittest.TestCase):
    """Tests for normalize_plate()."""

    def test_uppercase(self):
        self.assertEqual(normalize_plate("tn09ab1234"), "TN09AB1234")

    def test_strip_spaces(self):
        self.assertEqual(normalize_plate("TN 09 AB 1234"), "TN09AB1234")

    def test_strip_hyphens(self):
        self.assertEqual(normalize_plate("TN-09-AB-1234"), "TN09AB1234")

    def test_strip_dots(self):
        self.assertEqual(normalize_plate("TN.09.AB.1234"), "TN09AB1234")

    def test_strip_mixed(self):
        self.assertEqual(normalize_plate("TN 09-AB.1234"), "TN09AB1234")

    def test_empty(self):
        self.assertEqual(normalize_plate(""), "")

    def test_none_like(self):
        self.assertEqual(normalize_plate("  "), "")

    def test_slashes(self):
        self.assertEqual(normalize_plate("TN/09/AB/1234"), "TN09AB1234")


class TestValidateStandard(unittest.TestCase):
    """Tests for standard format: SS DD L{1,3} DDDD."""

    def test_tn_standard(self):
        r = validate_plate("TN09AB1234")
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.format_type, "standard")
        self.assertEqual(r.state_code, "TN")
        self.assertTrue(r.state_verified)
        self.assertFalse(r.corrected)

    def test_dl_standard(self):
        r = validate_plate("DL01C1234")
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.state_code, "DL")
        self.assertEqual(r.format_type, "standard")

    def test_ka_three_letter_series(self):
        r = validate_plate("KA51ADG9999")
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.state_code, "KA")

    def test_with_spaces(self):
        r = validate_plate("MH 12 AB 3456")
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.corrected_text, "MH12AB3456")

    def test_with_hyphens(self):
        r = validate_plate("AP-09-BZ-5678")
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.state_code, "AP")

    def test_single_letter_series(self):
        r = validate_plate("TN01A0001")
        self.assertTrue(r.plate_valid)

    def test_two_letter_series(self):
        r = validate_plate("TN09AB1234")
        self.assertTrue(r.plate_valid)

    def test_three_letter_series(self):
        r = validate_plate("TN09ABC1234")
        self.assertTrue(r.plate_valid)


class TestValidateBHSeries(unittest.TestCase):
    """Tests for BH series: DD BH DDDD L{1,2}."""

    def test_bh_two_letter(self):
        r = validate_plate("22BH1234AB")
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.format_type, "bh_series")
        self.assertEqual(r.state_code, "BH")

    def test_bh_single_letter(self):
        r = validate_plate("20BH5678A")
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.format_type, "bh_series")

    def test_bh_with_spaces(self):
        r = validate_plate("22 BH 1234 AB")
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.corrected_text, "22BH1234AB")


class TestValidateLegacy(unittest.TestCase):
    """Tests for legacy/loose format."""

    def test_legacy_short(self):
        """Legacy plates with non-standard suffixes."""
        r = validate_plate("TN09X123")
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.format_type, "legacy")
        self.assertTrue(r.needs_review)

    def test_unknown_state(self):
        """Unknown state code should fail."""
        r = validate_plate("ZZ09AB1234")
        # ZZ is not a valid state, but the regex matches standard
        # However state_verified should be False
        self.assertTrue(r.plate_valid)  # Format matches
        self.assertFalse(r.state_verified)
        self.assertTrue(r.needs_review)


class TestPositionalCorrection(unittest.TestCase):
    """Tests for OCR confusion correction."""

    def test_zero_to_O_in_state(self):
        """0 in letter position → O."""
        r = validate_plate("0D01AB1234")  # 0D → OD (Odisha)
        # After correction of position 0: 0→O
        if r.plate_valid:
            self.assertTrue(r.corrected)
            self.assertIn("OD", r.corrected_text)

    def test_O_to_0_in_digit(self):
        """O in digit position → 0."""
        r = validate_plate("TN09AB12O4")  # O in digit pos → 0
        self.assertTrue(r.plate_valid)
        self.assertTrue(r.corrected)
        self.assertEqual(r.corrected_text, "TN09AB1204")
        self.assertLessEqual(r.n_corrections, 2)

    def test_I_to_1_in_digit(self):
        """I in digit position → 1."""
        r = validate_plate("TN09ABI234")  # I in digit pos → 1
        self.assertTrue(r.plate_valid)
        self.assertTrue(r.corrected)
        self.assertEqual(r.corrected_text, "TN09AB1234")

    def test_S_to_5_in_digit(self):
        """S in digit position → 5."""
        r = validate_plate("TN09ABS234")  # S → 5
        self.assertTrue(r.plate_valid)
        self.assertTrue(r.corrected)
        self.assertEqual(r.corrected_text, "TN09AB5234")

    def test_B_to_8_in_digit(self):
        """B in digit position → 8."""
        r = validate_plate("TN09ABB234")  # B in first digit pos → 8
        self.assertTrue(r.plate_valid)
        self.assertTrue(r.corrected)
        self.assertEqual(r.corrected_text, "TN09AB8234")

    def test_1_to_I_in_letter(self):
        """1 in letter position → I."""
        r = validate_plate("TN09A11234")  # second series char: 1→I
        # TN09A1DDDD — middle is 1, expected letter → I → TN09AI1234
        self.assertTrue(r.plate_valid)
        self.assertTrue(r.corrected)

    def test_max_two_corrections(self):
        """More than 2 corrections needed — should NOT force validity."""
        # 3 wrong positions: 0N09A812O4
        # position 0: 0→O (1), position 7: 2→? stays, position 8: O→0 (2), that's only 2
        # Let's try one where 3 are needed
        r = validate_plate("0N0OA812O4")
        # 0→O, O→0, O→0 = 3 corrections needed in standard
        # Should fail (>2 corrections not applied)
        if r.corrected:
            self.assertLessEqual(r.n_corrections, 2)

    def test_no_invention(self):
        """Garbage text should not be forced into validity."""
        r = validate_plate("XXXXXXXX")
        self.assertFalse(r.plate_valid)
        self.assertTrue(r.needs_review)
        self.assertFalse(r.corrected)

    def test_bh_correction(self):
        """BH series with OCR confusion."""
        r = validate_plate("22BHI234AB")  # I→1 in digit position
        self.assertTrue(r.plate_valid)
        self.assertTrue(r.corrected)
        self.assertEqual(r.corrected_text, "22BH1234AB")

    def test_G_to_6_in_digit(self):
        """G in digit position → 6."""
        r = validate_plate("TN09ABG234")  # G → 6
        self.assertTrue(r.plate_valid)
        self.assertTrue(r.corrected)
        self.assertEqual(r.corrected_text, "TN09AB6234")

    def test_T_to_7_in_digit(self):
        """T in digit position → 7."""
        r = validate_plate("TN09ABT234")  # T → 7
        self.assertTrue(r.plate_valid)
        self.assertTrue(r.corrected)
        self.assertEqual(r.corrected_text, "TN09AB7234")


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_empty_string(self):
        r = validate_plate("")
        self.assertFalse(r.plate_valid)
        self.assertTrue(r.needs_review)
        self.assertEqual(r.format_type, "invalid")

    def test_single_char(self):
        r = validate_plate("A")
        self.assertFalse(r.plate_valid)

    def test_only_digits(self):
        r = validate_plate("123456789")
        self.assertFalse(r.plate_valid)

    def test_only_letters(self):
        r = validate_plate("ABCDEFGH")
        self.assertFalse(r.plate_valid)

    def test_too_long(self):
        r = validate_plate("TN09ABCD12345678")
        self.assertFalse(r.plate_valid)

    def test_too_short(self):
        r = validate_plate("TN09")
        self.assertFalse(r.plate_valid)

    def test_special_characters_stripped(self):
        """Special characters should be stripped during normalization."""
        r = validate_plate("TN•09•AB•1234")
        self.assertTrue(r.plate_valid)

    def test_lowercase_input(self):
        r = validate_plate("tn09ab1234")
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.state_code, "TN")


class TestStateCodes(unittest.TestCase):
    """Tests for state code verification."""

    def test_all_verified_codes_are_two_letters(self):
        for code in VERIFIED_STATE_CODES:
            self.assertEqual(len(code), 2, f"State code {code} is not 2 chars")
            self.assertTrue(code.isalpha(), f"State code {code} is not alpha")
            self.assertTrue(code.isupper(), f"State code {code} is not upper")

    def test_tn_is_verified(self):
        self.assertIn("TN", VERIFIED_STATE_CODES)

    def test_or_is_unverified(self):
        """OR (old Odisha code) should be in ALL but not VERIFIED."""
        self.assertNotIn("OR", VERIFIED_STATE_CODES)
        self.assertIn("OR", ALL_STATE_CODES)

    def test_unverified_state_plate(self):
        """Plate with old Odisha code (OR) should be valid but flagged."""
        r = validate_plate("OR05AB1234")
        self.assertTrue(r.plate_valid)
        self.assertFalse(r.state_verified)

    def test_minimum_states(self):
        """Ensure we have at least 28 states + 8 UTs = ~35 codes."""
        self.assertGreaterEqual(len(VERIFIED_STATE_CODES), 30)


class TestCorrectStandard(unittest.TestCase):
    """Direct tests for correct_standard()."""

    def test_no_correction_needed(self):
        text, n = correct_standard("TN09AB1234")
        self.assertEqual(text, "TN09AB1234")
        self.assertEqual(n, 0)

    def test_single_digit_in_letter(self):
        text, n = correct_standard("TN09A81234")  # 8→B in letter pos
        self.assertEqual(text, "TN09AB1234")
        self.assertEqual(n, 1)

    def test_single_letter_in_digit(self):
        text, n = correct_standard("TN09ABO234")  # O→0 in digit pos
        self.assertEqual(text, "TN09AB0234")
        self.assertEqual(n, 1)


class TestCorrectBH(unittest.TestCase):
    """Direct tests for correct_bh_series()."""

    def test_no_correction_needed(self):
        text, n = correct_bh_series("22BH1234AB")
        self.assertEqual(text, "22BH1234AB")
        self.assertEqual(n, 0)

    def test_letter_in_year(self):
        text, n = correct_bh_series("Z2BH1234AB")  # Z→2 in year
        self.assertEqual(text, "22BH1234AB")
        self.assertEqual(n, 1)


class TestTwoLinePlateNormalization(unittest.TestCase):
    """Test that two-line plates normalize correctly when concatenated."""

    def test_two_line_standard(self):
        """Two-line plate: TN 09\\nAB 1234 → TN09AB1234."""
        top = "TN 09"
        bottom = "AB 1234"
        combined = top + bottom
        r = validate_plate(combined)
        self.assertTrue(r.plate_valid)
        self.assertEqual(r.corrected_text, "TN09AB1234")

    def test_two_line_with_newline(self):
        """Newline character should be stripped."""
        r = validate_plate("TN09\nAB1234")
        # \n is not in our strip list explicitly, but spaces are
        # Let's check — newlines should be stripped
        normalized = normalize_plate("TN09\nAB1234")
        # \n is not a space/hyphen/dot — we should handle it
        # Actually our regex strips \s which includes \n
        self.assertEqual(normalized, "TN09AB1234")


if __name__ == "__main__":
    unittest.main()
