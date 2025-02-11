import pytz

import misc.constants as constants

from datetime import datetime, timedelta

gb_timezone = pytz.timezone('Europe/London')

def generate_settlement_dates(start_date_str, end_date_str, format_date_time_as_string = False):
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError("Incorrect date format, should be YYYY-MM-DD") from e

    date_list = [(start_date + timedelta(days=i)) for i in range((end_date - start_date).days + 1)]
    
    if format_date_time_as_string:
        date_list = [date.strftime('%Y-%m-%d') for date in date_list]
    
    return date_list

def get_settlement_dates_and_settlement_periods(start_date_str, end_date_str, convert_datetime_to_string):
    full_date_list = generate_settlement_dates(start_date_str, end_date_str)
    dates_with_settlement_periods_per_day = get_settlement_periods_for_each_day_in_date_range(full_date_list)
    if convert_datetime_to_string:
        dates_with_settlement_periods_per_day = {key.strftime('%Y-%m-%d'): value 
                                                 for key, value in dates_with_settlement_periods_per_day.items()}
    return dates_with_settlement_periods_per_day

def get_list_of_settlement_dates_and_periods(start_date, end_date):
    settlement_dates_with_periods_per_day = get_settlement_dates_and_settlement_periods(start_date, end_date, True)
    settlement_dates_and_periods = []
    for settlement_date, settlement_periods_in_day in settlement_dates_with_periods_per_day.items():
        for settlement_period in range(1, settlement_periods_in_day + 1):
            settlement_dates_and_periods.append(f"{settlement_date}-{settlement_period}")
            
    return settlement_dates_and_periods

def get_settlement_periods_for_each_day_in_date_range(settlement_dates_inclusive):
    settlement_periods_per_day = {}
    settlement_dates_for_calculation = settlement_dates_inclusive + [
        settlement_dates_inclusive[-1] + timedelta(days = 1)]

    for i in range(len(settlement_dates_for_calculation) - 1):
        current_date = settlement_dates_for_calculation[i]
        next_date = settlement_dates_for_calculation[i+1]
        offset_now = gb_timezone.utcoffset(current_date)
        offset_next = gb_timezone.utcoffset(next_date)
        settlement_periods_in_day = constants.DateTime.DEFAULT_SETTLEMENT_PERIODS_PER_DAY

        if offset_now != offset_next:
            settlement_periods_in_day = (constants.DateTime.SETTLEMENT_PERIODS_PER_SHORT_DAY if offset_next > offset_now 
                else constants.DateTime.SETTLEMENT_PERIODS_PER_LONG_DAY)
            
        settlement_periods_per_day[current_date] = settlement_periods_in_day
    
    return settlement_periods_per_day