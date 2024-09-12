from datetime import datetime
import numpy as np

def rename_cluster(cluster_id):
    if cluster_id in ['0', 0]:
        return 'Early Active'
    elif cluster_id in ['2', 2]:
        return 'Late Active'
    elif cluster_id in ['1', 1]:
        return 'Steady Active'
    elif cluster_id in ['3', 3]:
        return 'Steady Inactive'
    else:
        raise ValueError

def minmax_normalize(data):
    min_val = min(data)
    max_val = max(data)

    # Handle the case where all values in the list are the same
    if min_val == max_val:
        return [0.5] * len(data)

    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def to_timestamp(date_field, local_time=False) -> int:
    """
    Converts Given Time in the format of '2017-11-14 19:09:20 +1300' to timestamp.
    :param local_time: boolean indicates whether we want the local time or UTC time. It removes +0000 from the end
    :param date_field: string of date
    :return: timestamp of the give date in arguments
    :rtype: int
    """
    if not date_field:
        return False
    # 2020-12-28T22:46:14Z
    # If the input is in gitlog format (Wed Oct 18 13:05:28 2017 +1300), we change it to normal date (2017-10-30
    # 15:31:32 +1300)
    if len(date_field.split()) == 6:
        date_field = date_field.split()[4] + "-" + str(fetch_month_number(date_field.split()[1])) + "-" + \
                     date_field.split()[2] + " " + \
                     date_field.split()[3] + " " + date_field.split()[5]

    # if the input is 2015-03-24 we change it to 2015-03-24 00:00:00 +0000
    if len(date_field) == 10:
        date_field = date_field + " 00:00:00 +0000"

    # if the input is in github format 2020-12-28T22:46:14Z
    if len(date_field.split('T')) == 2:
        date_field = date_field.replace("Z", "")
        date_field = date_field.split('T')[0] + " " + date_field.split('T')[1] + " +0000"

    # COnvert the date into timestamp
    if local_time:
        date_field = date_field[:-6] + " +0000"

    # for 2014-06-24 08-44-05 +0000
    if len(date_field.split('-')) == 5:
        datetime_object = datetime.strptime(date_field, "%Y-%m-%d %H-%M-%S %z")
    else:
        datetime_object = datetime.strptime(date_field, "%Y-%m-%d %H:%M:%S %z")
    timestamp = datetime.timestamp(datetime_object)
    return int(timestamp)

def fetch_month_number(self):
    """
    From the given month name it returns the month number (e.g. 'Jan')
    :param self: string of month char
    :return: number of month
    :rtype: int
    """
    if self == 'Jan':
        return 1
    elif self == 'Feb':
        return 2
    elif self == 'Mar':
        return 3
    elif self == 'Apr':
        return 4
    elif self == 'May':
        return 5
    elif self == 'Jun':
        return 6
    elif self == 'Jul':
        return 7
    elif self == 'Aug':
        return 8
    elif self == 'Sep':
        return 9
    elif self == 'Oct':
        return 10
    elif self == 'Nov':
        return 11
    elif self == 'Dec':
        return 12
    else:
        raise Exception("not supported month string")

def categorize(input_list):
    import pandas as pd
    # Convert the list to a pandas Series
    data = pd.Series(input_list)

    # Calculate the quartiles
    q1 = data.quantile(0.25)
    q2 = data.quantile(0.50)
    q3 = data.quantile(0.75)

    # Function to categorize the numbers
    def categorize(value):
        if value <= q1:
            return "least"
        elif value <= q2:
            return "less"
        elif value <= q3:
            return "more"
        else:
            return "most"

    # Apply the categorization
    categories = data.apply(categorize)
    return categories.tolist()
