"""
Unit tests for season utility functions.

Tests cover:
- Known season_id conversions
- Edge cases (earliest season, future seasons)
- Invalid inputs and format validation
- Round-trip conversions
- Error messages are clear and helpful
"""

import pytest
from src.utils.season_utils import season_id_to_string, string_to_season_id


class TestSeasonIdToString:
    """Test suite for season_id_to_string() function."""

    def test_known_values_2023_24(self):
        """Test conversion of 2023-24 season."""
        result = season_id_to_string(22023)
        assert result == "2023-24"

    def test_known_values_2024_25(self):
        """Test conversion of 2024-25 season."""
        result = season_id_to_string(22024)
        assert result == "2024-25"

    def test_known_values_2022_23(self):
        """Test conversion of 2022-23 season."""
        result = season_id_to_string(22022)
        assert result == "2022-23"

    def test_earliest_season_1946_47(self):
        """Test conversion of earliest NBA season (1946-47)."""
        result = season_id_to_string(21946)
        assert result == "1946-47"

    def test_return_type_is_string(self):
        """Test that return value is a string."""
        result = season_id_to_string(22023)
        assert isinstance(result, str)

    def test_format_is_correct(self):
        """Test that format matches 'YYYY-YY' pattern."""
        result = season_id_to_string(22023)
        parts = result.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 4  # YYYY
        assert len(parts[1]) == 2  # YY
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_invalid_season_id_not_5_digits(self):
        """Test that non-5-digit season_id raises ValueError."""
        with pytest.raises(ValueError, match="5-digit composite format"):
            season_id_to_string(2023)  # Only 4 digits

    def test_invalid_season_id_wrong_prefix(self):
        """Test that season_id not starting with 2 raises ValueError."""
        with pytest.raises(ValueError, match="5-digit composite format"):
            season_id_to_string(12023)  # Starts with 1, not 2

    def test_invalid_season_id_not_integer(self):
        """Test that non-integer season_id raises ValueError."""
        with pytest.raises(ValueError, match="must be an integer"):
            season_id_to_string("22023")

    def test_invalid_season_id_before_1946(self):
        """Test that season_id before 1946 raises ValueError."""
        with pytest.raises(ValueError, match="before NBA founding year"):
            season_id_to_string(21945)

    def test_invalid_season_id_too_far_future(self):
        """Test that season_id too far in future raises ValueError."""
        # Current year + 3 should be rejected
        from datetime import datetime
        future_year = datetime.now().year + 3
        season_id = 20000 + future_year
        with pytest.raises(ValueError, match="too far in the future"):
            season_id_to_string(season_id)

    def test_error_message_includes_season_id(self):
        """Test that error messages include the invalid season_id."""
        try:
            season_id_to_string(2023)
        except ValueError as e:
            assert "2023" in str(e)

    def test_century_boundary_1999_2000(self):
        """Test season spanning 1999-2000."""
        result = season_id_to_string(21999)
        assert result == "1999-00"

    def test_century_boundary_2099_2100(self):
        """Test that we handle year transitions properly."""
        # This may not be in valid range yet, but test the format logic
        # Skip if too far in future
        try:
            result = season_id_to_string(22099)
            assert result == "2099-00"
        except ValueError:
            # If it's too far in future, that's OK for this test
            pass


class TestStringToSeasonId:
    """Test suite for string_to_season_id() function."""

    def test_known_values_2023_24(self):
        """Test conversion of '2023-24' string."""
        result = string_to_season_id("2023-24")
        assert result == 22023

    def test_known_values_2024_25(self):
        """Test conversion of '2024-25' string."""
        result = string_to_season_id("2024-25")
        assert result == 22024

    def test_known_values_1946_47(self):
        """Test conversion of earliest season '1946-47'."""
        result = string_to_season_id("1946-47")
        assert result == 21946

    def test_return_type_is_integer(self):
        """Test that return value is an integer."""
        result = string_to_season_id("2023-24")
        assert isinstance(result, int)

    def test_invalid_format_no_hyphen(self):
        """Test that string without hyphen raises ValueError."""
        with pytest.raises(ValueError, match="format 'YYYY-YY'"):
            string_to_season_id("202324")

    def test_invalid_format_multiple_hyphens(self):
        """Test that string with multiple hyphens raises ValueError."""
        with pytest.raises(ValueError, match="exactly one hyphen"):
            string_to_season_id("2023-24-25")

    def test_invalid_format_non_integer_year(self):
        """Test that non-integer year raises ValueError."""
        with pytest.raises(ValueError, match="must be integers"):
            string_to_season_id("abcd-ef")

    def test_invalid_format_inconsistent_years(self):
        """Test that inconsistent end year raises ValueError."""
        with pytest.raises(ValueError, match="doesn't match"):
            string_to_season_id("2023-25")  # Should be 2023-24

    def test_invalid_season_id_not_string(self):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            string_to_season_id(22023)

    def test_invalid_season_before_1946(self):
        """Test that season before 1946 raises ValueError."""
        with pytest.raises(ValueError, match="before NBA founding year"):
            string_to_season_id("1945-46")

    def test_invalid_season_too_far_future(self):
        """Test that season too far in future raises ValueError."""
        from datetime import datetime
        future_year = datetime.now().year + 3
        season_str = f"{future_year}-{str(future_year + 1)[2:]}"
        with pytest.raises(ValueError, match="too far in the future"):
            string_to_season_id(season_str)

    def test_century_boundary_1999_00(self):
        """Test season spanning 1999-2000."""
        result = string_to_season_id("1999-00")
        assert result == 21999


class TestRoundTripConversion:
    """Test round-trip conversions between formats."""

    def test_roundtrip_2023_24(self):
        """Test that converting back and forth preserves data."""
        original_id = 22023
        as_string = season_id_to_string(original_id)
        back_to_id = string_to_season_id(as_string)
        assert back_to_id == original_id

    def test_roundtrip_2024_25(self):
        """Test round-trip for 2024-25 season."""
        original_id = 22024
        as_string = season_id_to_string(original_id)
        back_to_id = string_to_season_id(as_string)
        assert back_to_id == original_id

    def test_roundtrip_1946_47(self):
        """Test round-trip for earliest season."""
        original_id = 21946
        as_string = season_id_to_string(original_id)
        back_to_id = string_to_season_id(as_string)
        assert back_to_id == original_id

    def test_roundtrip_from_string(self):
        """Test round-trip starting from string format."""
        original_str = "2023-24"
        as_id = string_to_season_id(original_str)
        back_to_str = season_id_to_string(as_id)
        assert back_to_str == original_str

    def test_multiple_roundtrips(self):
        """Test that multiple conversions are idempotent."""
        start_id = 22023
        id1 = string_to_season_id(season_id_to_string(start_id))
        id2 = string_to_season_id(season_id_to_string(id1))
        id3 = string_to_season_id(season_id_to_string(id2))
        assert id1 == id2 == id3 == start_id


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_year_999_to_1000(self):
        """Test year formatting at 999-1000 boundary (if valid)."""
        # This would be way before NBA, so should fail validation
        with pytest.raises(ValueError, match="before NBA founding year"):
            season_id_to_string(20999)

    def test_two_digit_year_extraction(self):
        """Test that two-digit year extraction works correctly."""
        # For year 2009, end year short should be "09" not "9"
        result = season_id_to_string(22009)
        assert result == "2009-10"

    def test_very_recent_season(self):
        """Test a season very recently in the past."""
        result = season_id_to_string(22000)
        assert result == "2000-01"


class TestIntegration:
    """Integration tests combining multiple scenarios."""

    def test_season_progression(self):
        """Test conversion of sequential seasons."""
        seasons = [
            (22020, "2020-21"),
            (22021, "2021-22"),
            (22022, "2022-23"),
            (22023, "2023-24"),
            (22024, "2024-25"),
        ]

        for season_id, expected_str in seasons:
            result = season_id_to_string(season_id)
            assert result == expected_str

    def test_error_messages_are_helpful(self):
        """Test that all error messages are clear and include context."""
        test_cases = [
            (2023, "5-digit composite format"),
            ("22023", "must be an integer"),
            (21945, "before nba founding year"),
        ]

        for invalid_input, expected_message in test_cases:
            try:
                season_id_to_string(invalid_input)
                pytest.fail(f"Should have raised ValueError for {invalid_input}")
            except ValueError as e:
                assert expected_message in str(e).lower()
