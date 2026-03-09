import pandas as pd
import numpy as np

def generic_load(load_type, load_previous_year_type, peakmonth, daysInMonth, user_defined_load):
    if load_type == 8 or load_previous_year_type == 9:
        if peakmonth == 'July':
            path_generic_load = 'content/Generic_load_JulyP.csv'
        else:
            path_generic_load = 'content/Generic_load_JanuaryP.csv'

        EloadData = pd.read_csv(path_generic_load, header=None).values
        Eload = np.array(EloadData[:, 0])
        Eload_total = np.sum(Eload)
        scaling_factor = user_defined_load / Eload_total
        scaled_data = scaling_factor * Eload
    elif load_type == 9 or load_previous_year_type == 10:
        if peakmonth == 'July':
            path_generic_load = 'content/Generic_load_JulyP.csv'
        else:
            path_generic_load = 'content/Generic_load_JanuaryP.csv'

        EloadData = pd.read_csv(path_generic_load, header=None).values
        Eload = np.array(EloadData[:, 0])
        scaled_data = Eload
    else:
        if peakmonth == 'July':
            path_generic_load = 'content/Generic_load_JulyP.csv'
        else:
            path_generic_load = 'content/Generic_load_JanuaryP.csv'

        EloadData = pd.read_csv(path_generic_load, header=None).values
        Eload = np.array(EloadData[:, 0])

        # Initialize a DataFrame to store scaled data
        scaled_data = pd.DataFrame()

        # Split the data into months based on days in each month
        start = 0
        for month, days in enumerate(daysInMonth):
            end = start + 24 * days  # Assuming 24 hours in a day
            actual_load = Eload[start:end]

            # Calculate the scaling factor
            scaling_factor = user_defined_load[month] / np.sum(actual_load)

            # Scale the data for the current month and convert it to a Series
            scaled_month_data = pd.Series(actual_load * scaling_factor)

            # Append scaled data to the result DataFrame
            scaled_data = pd.concat([scaled_data, scaled_month_data], axis=0)

            start = end
        scaled_data = scaled_data.values.flatten()
    return scaled_data