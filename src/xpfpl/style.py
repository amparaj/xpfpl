"""Club colours for the dashboard, shared by app.py and the Markets tab."""

import pandas as pd

CLUB_COLOURS = {
    "ARS": "#EF0107", "AVL": "#670E36", "BOU": "#DA291C", "BRE": "#E30613",
    "BHA": "#0057B8", "BUR": "#6C1D45", "CHE": "#034694", "CRY": "#1B458F",
    "COV": "#59CBE8", "EVE": "#003399", "FUL": "#000000", "HUL": "#F5A12D",
    "IPS": "#0044A9", "LEE": "#FFCD00", "LIV": "#C8102E",
    "MCI": "#6CABDD", "MUN": "#DA291C", "NEW": "#241F20", "NFO": "#DD0000",
    "SUN": "#EB172B", "TOT": "#132257", "WHU": "#7A263A", "WOL": "#FDB913",
}
CLUB_TEXT = {c: "#111111" if c in ("COV", "HUL", "LEE", "MCI", "WOL") else "#ffffff"  # readable on each
             for c in CLUB_COLOURS}
NEUTRAL_BG, NEUTRAL_FG = "#d9dde3", "#111111"

# Number formats for st.column_config.NumberColumn, so every table shows the same kind of number the same way.
# Set one on every float column of a Styler-wrapped table: without it pandas shows six decimals ("15.500000").
POINTS = "%.2f"        # points and xP: always two decimals
PERCENT = "percent"    # a 0-1 chance -> "35.5%": at most two decimals, trailing zeros dropped ("100%")
DECIMAL = "localized"  # anything else, including 0-100 "... %" columns: at most three decimals, zeros dropped


def club_cell(short_name: str) -> str:
    """CSS for a cell showing a club's short name, in that club's colours."""
    return (f"background-color: {CLUB_COLOURS.get(short_name, NEUTRAL_BG)}; "
            f"color: {CLUB_TEXT.get(short_name, NEUTRAL_FG)}")


def club_columns(frame: pd.DataFrame, columns: tuple[str, ...] = ("Club",), yes_no: tuple[str, ...] = ()):
    """A Styler colouring every column in `columns` (whose values are club short names), and every
    column in `yes_no` green where it says "Yes..." and red where it says "No"."""
    def style(column: pd.Series) -> list[str]:
        if column.name in yes_no:
            return [yes_no_cell(v) for v in column]
        if column.name not in columns:
            return [""] * len(column)
        return [club_cell(v) if isinstance(v, str) else "" for v in column]
    return frame.style.apply(style, axis=0)


def yes_no_cell(value) -> str:
    """CSS for a "Yes"/"No" answer: green for yes (including "Yes (2)"), red for no, anything else plain."""
    if isinstance(value, str) and value.startswith("Yes"):
        return f"background-color: {YES_BG}; color: {YES_FG}"
    if value == "No":
        return f"background-color: {NO_BG}; color: {NO_FG}"
    return ""


YES_BG, YES_FG = "#c9ecd3", "#0f5323"
NO_BG, NO_FG = "#f8cfcf", "#8a1414"
