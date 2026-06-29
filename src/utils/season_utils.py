"""
NBA Season Utilities
====================

Handles conversion and validation of NBA season identifiers.

Season ID Format:
- Internal format: 5-digit composite integer (2YYYY)
  Example: 22023 represents the 2023-24 season
  Structure: '2' prefix + 4-digit year (start year)

- nba_api format: "YYYY-YY" string
  Example: "2023-24"
  Structure: start_year + "-" + last two digits of end year

Author: NBA Prediction Project
Date: 2024
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Valid NBA season range
EARLIEST_SEASON_YEAR = 1946  # NBA founded in 1946
CURRENT_YEAR = datetime.now().year


def season_id_to_string(season_id: int) -> str:
    """
    Convert NBA season_id to nba_api 'YYYY-YY' format.

    NBA internally stores season_id as a composite: 2 + start_year.
    Example: 22023 = 2023-24 season (2 + 2023)

    nba_api expects format: "YYYY-YY"
    Example: "2023-24"

    Args:
        season_id: int like 22023 (composite format: 2 + year)

    Returns:
        str like "2023-24" (format: "YYYY-YY")

    Raises:
        ValueError: if season_id format is invalid or out of range

    Examples:
        >>> season_id_to_string(22023)
        '2023-24'
        >>> season_id_to_string(22024)
        '2024-25'
        >>> season_id_to_string(21946)
        '1946-47'
    """
    # Validate season_id is an integer
    if not isinstance(season_id, int):
        raise ValueError(
            f"season_id must be an integer, got {type(season_id).__name__}"
        )

    # Validate season_id format (should be 5 digits starting with 2)
    season_id_str = str(season_id)
    if len(season_id_str) != 5 or not season_id_str.startswith("2"):
        raise ValueError(
            f"season_id must be 5-digit composite format (2YYYY), got {season_id}"
        )

    # Extract year from season_id
    year = season_id % 10000

    # Validate year is in reasonable range
    if year < EARLIEST_SEASON_YEAR:
        raise ValueError(
            f"season_id year {year} is before NBA founding year {EARLIEST_SEASON_YEAR}"
        )

    # Allow up to 2 years in the future for preseason planning
    if year > CURRENT_YEAR + 2:
        raise ValueError(
            f"season_id year {year} is too far in the future (current year: {CURRENT_YEAR})"
        )

    # Format as "YYYY-YY"
    end_year = year + 1
    end_year_short = str(end_year)[2:]  # Get last 2 digits
    formatted_season = f"{year}-{end_year_short}"

    logger.debug(f"Converted season_id {season_id} to {formatted_season}")
    return formatted_season


def string_to_season_id(season_str: str) -> int:
    """
    Convert nba_api 'YYYY-YY' format to NBA season_id.

    Inverse of season_id_to_string.

    Args:
        season_str: str like "2023-24"

    Returns:
        int like 22023 (composite format: 2 + year)

    Raises:
        ValueError: if season_str format is invalid

    Examples:
        >>> string_to_season_id("2023-24")
        22023
        >>> string_to_season_id("1946-47")
        21946
    """
    if not isinstance(season_str, str):
        raise ValueError(
            f"season_str must be a string, got {type(season_str).__name__}"
        )

    # Validate format "YYYY-YY"
    if "-" not in season_str:
        raise ValueError(
            f"season_str must be in format 'YYYY-YY', got '{season_str}'"
        )

    parts = season_str.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"season_str must have exactly one hyphen, got '{season_str}'"
        )

    try:
        year = int(parts[0])
        end_year_short = int(parts[1])
    except ValueError:
        raise ValueError(
            f"season_str parts must be integers, got '{season_str}'"
        )

    # Validate consistency
    expected_end_year_short = (year + 1) % 100
    if end_year_short != expected_end_year_short:
        raise ValueError(
            f"Invalid season_str '{season_str}': end year {end_year_short} "
            f"doesn't match {year} + 1 = {year + 1}"
        )

    # Validate year is in range
    if year < EARLIEST_SEASON_YEAR:
        raise ValueError(
            f"season_str year {year} is before NBA founding year {EARLIEST_SEASON_YEAR}"
        )

    if year > CURRENT_YEAR + 2:
        raise ValueError(
            f"season_str year {year} is too far in the future (current year: {CURRENT_YEAR})"
        )

    # Convert to season_id (2 + year)
    season_id = 20000 + year
    logger.debug(f"Converted season_str {season_str} to season_id {season_id}")
    return season_id
