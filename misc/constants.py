from enum import Enum
from datetime import datetime

class ColumnHeaders(Enum):
    DATE_PERIOD_PRIMARY_KEY = 'settlement_date_and_period'
    NET_IMBALANCE_VOLUME = 'net_imbalance_volume'
    
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
    
class DatabaseNames(Enum):
    NIV_CHASING = 'niv_chasing'
    SYSTEM_PROPERTIES = 'system_properties'
    PRICES = 'prices'

class TableNames(Enum):
    BID_OFFER = 'bid_offer_data'
    BMUS = 'bm_unit_information'
    BOA = 'boa_data'
    BSC = 'elexon_bsc_roles'
    DA_WIND_FORECAST = 'day_ahead_wind_forecast'
    DERATED_MARGIN_fORECAST = 'derated_margin_forecast'
    FORECAST_ERRORS = 'forecast_errors'
    GENERATION = 'generation_data'
    HISTORICAL_PRICES = 'historical_prices'
    INDICATED_FORECASTS = 'indicated_forecasts'
    ITSDO = 'initial_transmission_system_demand_outturn'
    LAGGED_SYSTEM_PRICES = 'lagged_system_prices'
    MR1B = 'mr1b_report_data'
    NIV_FORECAST_TRAINING_DATA = 'niv_forecast_training_data'
    OUTAGES = 'outages'
    PRICES = 'prices'
    SYSTEM_IMBALANCE = 'system_imbalance_data'
    TEST = 'test'
    TLMS = 'tlm_data'
    TSDF = 'transmission_system_demand_forecast'