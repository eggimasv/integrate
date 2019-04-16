
import csv
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

def get_month_from_string(month_string):
    """Convert string month to int month with Jan == 1

    Arguments
    ----------
    month_string : str
        Month given as a string

    Returns
    --------
    month : int
        Month as an integer (jan = 1, dez = 12)
    """
    if month_string == 'Jan':
        month = 1
    elif month_string == 'Feb':
        month = 2
    elif month_string == 'Mar':
        month = 3
    elif month_string == 'Apr':
        month = 4
    elif month_string == 'May':
        month = 5
    elif month_string == 'Jun':
        month = 6
    elif month_string == 'Jul':
        month = 7
    elif month_string == 'Aug':
        month = 8
    elif month_string == 'Sep':
        month = 9
    elif month_string == 'Oct':
        month = 10
    elif month_string == 'Nov':
        month = 11
    elif month_string == 'Dec':
        month = 12

    return int(month)
def date_to_yearday(year, month, day):
    """Gets the yearday (julian year day) of a year minus one to correct because of python iteration

    Arguments
    ----------
    date_base_yr : int
        Year
    date_base_yr : int
        Month
    day : int
        Day

    Example
    -------
    5. January 2015 --> Day nr 5 in year --> -1 because of python --> Out: 4
    """
    date_y = date(year, month, day)
    yearday = date_y.timetuple().tm_yday - 1 #: correct because of python iterations

    return yearday

def mw_to_gwh(megawatt, number_of_hours):
    """"Conversion of MW to GWh

    Arguments
    ---------
    kwh : float
        Kilowatthours
    number_of_hours : float
        Number of hours

    Return
    ------
    gwh : float
        Gigawatthours

    """
    # Convert MW to MWh
    megawatt_hour = megawatt * number_of_hours

    # Convert mwth to gwh
    gigawatthour = megawatt_hour / 1000.0

    return gigawatthour

def read_raw_elec_2015(path_to_csv, year=2015):
    """Read in national electricity values provided
    in MW and convert to GWh

    Arguments
    ---------
    path_to_csv : str
        Path to csv file
    year : int
        Year of data

    Returns
    -------
    elec_data_indo : array
        Hourly INDO electricity in GWh (INDO - National Demand)
    elec_data_itsdo : array
        Hourly ITSDO electricity in GWh (Transmission System Demand)

    Note
    -----
    Half hourly measurements are aggregated to hourly values

    Necessary data preparation: On 29 March and 25 Octobre
    there are 46 and 48 values because of the changing of the clocks
    The 25 Octobre value is omitted, the 29 March hour interpolated
    in the csv file

    Source
    ------
    http://www2.nationalgrid.com/uk/Industry-information/electricity-transmission-operational-data/
    For more information on INDO and ISTDO see DemandData Field Descriptions file:
    http://www2.nationalgrid.com/WorkArea/DownloadAsset.aspx?id=8589934632

    National Demand is calculated as a sum
    of generation based on National Grid
    operational generation metering
    """
    elec_data_indo = np.zeros((365, 24), dtype="float")
    elec_data_itsdo = np.zeros((365, 24), dtype="float")

    with open(path_to_csv, 'r') as csvfile:
        read_lines = csv.reader(csvfile, delimiter=',')
        _headings = next(read_lines)

        hour = 0
        counter_half_hour = 0

        for line in read_lines:
            month = get_month_from_string(line[0].split("-")[1])
            day = int(line[0].split("-")[0])

            # Get yearday
            yearday = date_to_yearday(year, month, day)

            if counter_half_hour == 1:
                counter_half_hour = 0

                # Sum value of first and second half hour
                hour_elec_demand_INDO = half_hour_demand_indo + float(line[2])
                hour_elec_demand_ITSDO = half_hour_demand_itsdo + float(line[4])

                # Convert MW to GWH (input is MW aggregated for two half
                # hourly measurements, therfore divide by 0.5)
                elec_data_indo[yearday][hour] = mw_to_gwh(hour_elec_demand_INDO, 0.5)
                elec_data_itsdo[yearday][hour] = mw_to_gwh(hour_elec_demand_ITSDO, 0.5)

                hour += 1
            else:
                counter_half_hour += 1

                half_hour_demand_indo = float(line[2]) # INDO - National Demand
                half_hour_demand_itsdo = float(line[4]) # Transmission System Demand

            if hour == 24:
                hour = 0

    return elec_data_indo, elec_data_itsdo

# load 2015 elec data 
path_elec_2015 = "C:/Users/cenv0553/integrate/data/elec_demand_2015.csv"
df_data = pd.read_csv(path_elec_2015)

national_demand, _  = read_raw_elec_2015(path_elec_2015)
print("Shape: " + str(national_demand.shape))


# Convert to dataframe
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.line.html
hours = range(8760)
demand = national_demand.reshape((8760))
df = pd.DataFrame(
    {'demand': demand},
    index=hours)
print("Lenth of dataframe: " + str(len(df)))

# PLot lines df.plot(kind='scatter',x='num_children',y='num_pets',color='red')
'''lines = df.plot.line(
    style="-",
    lw=0.5,
    xlim=[0, 8760],
    color='b'
    )'''

# ----------------------------------
# ACROSS THE WHOLE SIMULATION PERIOD
# ----------------------------------
selection_start = 0
selection_end = 1200 # 50 days

x_values_all = np.array(df.index)
y_values_all = np.array(df['demand'])

x_values = x_values_all[list(range(selection_start, selection_end))]
y_values = y_values_all[list(range(selection_start, selection_end))]

'''plt.plot(
    x_values,
    y_values,
    color='grey')'''

# Fill below line #https://stackoverflow.com/questions/16417496/matplotlib-fill-between-multiple-lines
plt.fill_between(x_values, y_values, facecolor='grey', alpha=0.5)

#plt.show()

# ----------------------------------
# Selection moving window
# ----------------------------------
 #[i * 24 for i in range(30)] #list(range(100, 120)) + list(range(100, 120))
selections = [
    list(range(10, 20)),
    list(range(130, 140))]

for x_selection in selections:
    y_selection = y_values_all[x_selection]

    plt.fill_between(x_selection, y_selection, facecolor='red', alpha=0.5)

    # Cut window
    plt.xlim(selection_start, selection_end)
    plt.ylim(20, 55)

plt.show()

print("AAAAAAA")





