import re

PLACEHOLDER_PATTERN = re.compile(r"<[^>]*>")
INT_PATTERN = re.compile(r"[+-]?(\d+)")
PREFIXED_RELATION_ENUM = ("op", "snt")
PREFIXED_RELATION_PREP = "prep"
PREFIXED_RELATION_PATTERN = re.compile(r"(?:(op|snt)\d+|(prep)-\w+)(-of)?")
PREFIXED_RELATION_SUBSTITUTION = r"\1\2\3"

NUM_PATTERN = re.compile(r"[+-]?\d+(\.\d+)?")
TOKEN_PLACEHOLDER = "<t>"
TOKEN_TITLE_PLACEHOLDER = "<T>"
LEMMA_PLACEHOLDER = "<l>"

# Specific relations
POLARITY = "polarity"
NAME = "name"
OP = "op"
MODE = "mode"
ARG2 = "arg2"
VALUE = "value"
DAY = "day"
MONTH = "month"
YEAR = "year"
YEAR2 = "year2"
DECADE = "decade"
WEEKDAY = "weekday"
QUARTER = "quarter"
CENTURY = "century"
SEASON = "season"
TIMEZONE = "timezone"

# Specific node property values
MINUS = "-"
MODES = ("expressive", "imperative", "interrogative")
DATE_ENTITY = "date-entity"

MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
SEASONS = ("winter", "fall", "spring", "summer")

UNRESOLVED = (NAME, POLARITY, OP, MODE) + MODES
