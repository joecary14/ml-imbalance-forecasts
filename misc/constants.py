from enum import Enum
from datetime import datetime

class ColumnHeaders(Enum):
    DATE_PERIOD_PRIMARY_KEY = 'settlement_date_and_period'
    
class DateTime:
    DAYS_PER_WEEK = 7
    DEFAULT_DATETIME = datetime.fromisoformat('1900-01-01T00:00:00')
    DEFAULT_TIME_STAMP = " 00:00:00"
    DEFAULT_SETTLEMENT_PERIODS_PER_DAY = 48
    MINUTES_PER_HOUR = 60
    MINUTES_PER_HALF_HOUR = 30
    MONTHS_PER_YEAR = 12
    SETTLEMENT_PERIODS_PER_SHORT_DAY = 46
    SETTLEMENT_PERIODS_PER_LONG_DAY = 50
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 3600