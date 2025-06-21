

import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import datetime

def verify_nmea_checksum(row):
    """
    Verifies the NMEA-style checksum for a row.
    Args:
        row (str): The full row as a string, e.g. "$PIXSE,DEPIN_,1.23,456.7*4A"
    Returns:
        bool: True if checksum matches, False otherwise.
        int: Calculated checksum.
        int: Expected checksum.
    """
    if not row.startswith('$') or '*' not in row:
        return False, None, None

    data, checksum_str = row[1:].split('*', 1)
    calc_checksum = 0
    for char in data:
        calc_checksum ^= ord(char)
    try:
        expected_checksum = int(checksum_str[:2], 16)
    except ValueError:
        return False, calc_checksum, None

    return calc_checksum == expected_checksum, calc_checksum, expected_checksum

def read_unique_identifiers(file_path):
    # Dictionary to store data for each identifier as DataFrames
    # date is in the file path 
    match = re.search(r'phinsdata_(\d{8})', os.path.basename(file_path))
    if match:
        date = match.group(1)
    else:
        raise ValueError(f"Could not extract date from file path: {file_path}")

    data_frames = {
        'DEPIN_': {'Depth': [], 'Time': [], 'Time_REF': []},
        'GPSIN_': {'Latitude': [], 'Longitude': [], 'Altitude': [], 'Time': [], 'Quality': [], 'Time_REF': []},
        'LOGIN_': {'XS1': [], 'XS2': [], 'XS3': [], 'HeadingMis': [], 'Time': [], 'Time_REF': []},
        'STDSPD': {'NorthSpeed': [], 'EastSpeed': [], 'VerticalSpeed': [], 'Time_REF': []},
        'STDPOS': {'Latitude': [], 'Longitude': [], 'Altitude': [], 'Time_REF': []},
        'TIME__': {'Time_REF': []},
        'ATITUD': {'Roll': [], 'Pitch': [], 'Time_REF': []},
        'HT_STS': {f'Bit{i}': [] for i in range(8)},
        'ALGSTS': {'Time_REF': [], **{f'Bit{i}': [] for i in range(8)}},
        'UTMWGS': {'LatZone': [], 'LonZone': [], 'Easting': [], 'Northing': [], 'Altitude': [], 'Time_REF': []},
        'HEAVE_': {'Surge': [], 'Sway': [], 'Heave': [], 'Time_REF': []},
        'SPEED_': {'EastSpeed': [], 'NorthSpeed': [], 'UpSpeed': [], 'Velocity': [], 'Time_REF': []},
        'POSITI': {'Latitude': [], 'Longitude': [], 'Altitude': [], 'Time_REF': []},
        'UTCIN_': {'Time': [], 'Time_REF': []},
        'STDHRP': {'HeadingSTD': [], 'RollSTD': [], 'PitchSTD': [], 'Time_REF': []},
        'LOGDVL': {'SoundVelocity': [], 'Compensation': [], 'DVL_Distance_2btm': [], 'Time_REF': []},
        'HEHDT': {'Heading': [], 'Time_REF': []},
        'HETHS': {'Heading': [], 'Mode': [], 'Time_REF': []},
    }
    # time updates in a separate data string, add to all the other data frames based on this one
    TIME=""
    Time_Last=0
    # Open the binary file and decode it as text
    with open(file_path, 'rb') as binary_file:
        decoded_content = binary_file.read().replace(b'\x00', b'').decode('utf-8')

        # Use csv.reader to parse the decoded content
        csv_reader = csv.reader(decoded_content.splitlines())

        for row in csv_reader:
            row_str = ','.join(row)  # Join columns back to a single string
            is_valid, calc, expected = verify_nmea_checksum(row_str)
            if not is_valid:
                print(f"Invalid checksum for row: {row}, calculated: {calc}, expected: {expected}")
                continue
            row = [col.split('*')[0] for col in row]
            if len(row) >= 2:  # Ensure there's at least one identifier
                identifier = row[0]
                if identifier == '$PIXSE':
                    identifier2 = row[1]
                    if identifier2 == 'DEPIN_' and len(row) >= 4:
                        if identifier2 == 'DEPIN_' and len(row) >= 4:
                            data_frames['DEPIN_']['Depth'].append(float(row[2]))
                            data_frames['DEPIN_']['Time'].append(float(row[3]))
                            data_frames['DEPIN_']['Time_REF'].append(TIME)
                    elif identifier2 == 'GPSIN_' and len(row) >= 7:
                        if float(row[2]) == 0 and float(row[3]) == 0:
                            continue
                        data_frames['GPSIN_']['Latitude'].append(float(row[2]))
                        data_frames['GPSIN_']['Longitude'].append(float(row[3]))
                        data_frames['GPSIN_']['Altitude'].append(float(row[4]))
                        data_frames['GPSIN_']['Time'].append(float(row[5]))
                        data_frames['GPSIN_']['Quality'].append(int(row[6]))
                        # 0 and ≥ 5: Fix not valid
                        # 1: GPS SPS Mode Fix not valid
                        # 2: Differential Mode, SPS Mode,  Fix not valid
                        # 3: GPS PPS Mode, Fix not valid
                        # 4: GPS RTK Mode, Fix not valid  
                        data_frames['GPSIN_']['Time_REF'].append(TIME)
                    elif identifier2 == 'LOGIN_' and len(row) >= 7:
                        data_frames['LOGIN_']['XS1'].append(float(row[2]))
                        data_frames['LOGIN_']['XS2'].append(float(row[3]))
                        data_frames['LOGIN_']['XS3'].append(float(row[4]))
                        data_frames['LOGIN_']['HeadingMis'].append(float(row[5]))
                        data_frames['LOGIN_']['Time'].append(float(row[6]))
                        data_frames['LOGIN_']['Time_REF'].append(TIME)
                    elif identifier2 == 'STDSPD' and len(row) >= 6:
                        data_frames['STDSPD']['NorthSpeed'].append(float(row[2]))
                        data_frames['STDSPD']['EastSpeed'].append(float(row[3]))
                        data_frames['STDSPD']['VerticalSpeed'].append(float(row[4]))
                        data_frames['STDSPD']['Time_REF'].append(TIME)
                    elif identifier2 == 'STDPOS' and len(row) >= 5:
                        data_frames['STDPOS']['Latitude'].append(float(row[2]))
                        data_frames['STDPOS']['Longitude'].append(float(row[3]))
                        data_frames['STDPOS']['Altitude'].append(float(row[4]))
                        data_frames['STDPOS']['Time_REF'].append(TIME)
                    elif identifier2 == 'TIME__' and len(row) >= 3:
                        Time_=float(row[2])
                        # skip 0 because it is not a valid time
                        if Time_ == 0:
                            continue
                        # if time crosses midnight, add a day to the date
                        if Time_ < Time_Last and Time_Last> 230000:
                            date_dt = datetime.datetime.strptime(date, "%Y%m%d")
                            date_dt += datetime.timedelta(days=1)
                            date = date_dt.strftime("%Y%m%d")
                        Time_Last =Time_
                        TIME=date+" "+row[2]
                        data_frames['TIME__']['Time_REF'].append(TIME)
                    elif identifier2 == 'ATITUD' and len(row) >= 4:
                        data_frames['ATITUD']['Roll'].append(float(row[2]))
                        data_frames['ATITUD']['Pitch'].append(float(row[3]))
                        data_frames['ATITUD']['Time_REF'].append(TIME)
                    elif identifier2 == 'HT_STS' and len(row) >= 3:
                        status = row[2]
                        for i in range(8):
                            data_frames['HT_STS'][f'Bit{i}'].append(status[i])
                    elif identifier2 == 'ALGSTS' and len(row) >= 3:
                        status = row[2]
                        data_frames['ALGSTS']['Time_REF'].append(TIME)
                        for i in range(8):
                            data_frames['ALGSTS'][f'Bit{i}'].append(status[i])
                    elif identifier2 == 'UTMWGS' and len(row) >= 7:
                        data_frames['UTMWGS']['LatZone'].append(row[2])
                        data_frames['UTMWGS']['LonZone'].append(int(row[3]))
                        data_frames['UTMWGS']['Easting'].append(float(row[4]))
                        data_frames['UTMWGS']['Northing'].append(float(row[5]))
                        data_frames['UTMWGS']['Altitude'].append(float(row[6]))
                        data_frames['UTMWGS']['Time_REF'].append(TIME)
                    elif identifier2 == 'HEAVE_' and len(row) >= 5:
                        data_frames['HEAVE_']['Surge'].append(float(row[2]))
                        data_frames['HEAVE_']['Sway'].append(float(row[3]))
                        data_frames['HEAVE_']['Heave'].append(float(row[4]))
                        data_frames['HEAVE_']['Time_REF'].append(TIME)
                    elif identifier2 == 'SPEED_' and len(row) >= 5:
                        data_frames['SPEED_']['EastSpeed'].append(float(row[2]))
                        data_frames['SPEED_']['NorthSpeed'].append(float(row[3]))
                        data_frames['SPEED_']['UpSpeed'].append(float(row[4]))
                        data_frames['SPEED_']['Velocity'].append((float(row[2])**2+float(row[3])**2)**0.5)
                        data_frames['SPEED_']['Time_REF'].append(TIME)
                    elif identifier2 == 'POSITI' and len(row) >= 5:
                        data_frames['POSITI']['Latitude'].append(float(row[2]))
                        data_frames['POSITI']['Longitude'].append(float(row[3]))
                        data_frames['POSITI']['Altitude'].append(float(row[4]))
                        data_frames['POSITI']['Time_REF'].append(TIME)
                    elif identifier2 == 'UTCIN_' and len(row) >= 2:
                        data_frames['UTCIN_']['Time'].append(float(row[2]))
                        data_frames['UTCIN_']['Time_REF'].append(TIME)
                    elif identifier2 == 'STDHRP' and len(row) >= 5:
                        data_frames['STDHRP']['HeadingSTD'].append(float(row[2]))
                        data_frames['STDHRP']['RollSTD'].append(float(row[3]))
                        data_frames['STDHRP']['PitchSTD'].append(float(row[4]))
                        data_frames['STDHRP']['Time_REF'].append(TIME)
                    elif identifier2 == 'LOGDVL' and len(row) >= 5:
                        if float(row[2]) > 0:
                            data_frames['LOGDVL']['SoundVelocity'].append(float(row[2]))
                            data_frames['LOGDVL']['Compensation'].append(float(row[3]))
                            data_frames['LOGDVL']['DVL_Distance_2btm'].append(float(row[4]))
                            data_frames['LOGDVL']['Time_REF'].append(TIME)
                elif identifier == '$HEHDT' and len(row) >= 2:
                    try :
                        data_frames['HEHDT']['Heading'].append(float(row[1]))
                        data_frames['HEHDT']['Time_REF'].append(TIME)
                    except ValueError:
                        a=1
                elif identifier == '$HETHS' and len(row) >= 3:
                    try:
                        data_frames['HETHS']['Heading'].append(float(row[1]))
                        data_frames['HETHS']['Mode'].append(row[2])
                        data_frames['HETHS']['Time_REF'].append(TIME)
                    except ValueError:
                        a=1

    # Convert each dictionary to a DataFrame
    for key in data_frames:
        data_frames[key] = pd.DataFrame(data_frames[key])
        if 'Time_REF' in data_frames[key].columns:
            # print(f"Converting {key} Time column to datetime")
            # print(data_frames[key])
            data_frames[key]['Date_Time'] = pd.to_datetime(data_frames[key]['Time_REF'], format='%Y%m%d %H%M%S.%f', errors='coerce')
            # remove the Time_REF column
            data_frames[key].drop(columns=['Time_REF'], inplace=True)
            # print(data_frames[key]['Date_Time'])

    
    return data_frames

def plot_Phins_Data(data_frames,Start,End,Savefile=None):
    # Plotting the data based on start and end 
    # if 0 plot all aviable data
    

    # Figures_Dir = os.path.join(Data_Dir, "Figures")
    # os.makedirs(Figures_Dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    # Plotting Depth
    if 'DEPIN_' in data_frames:
        ax.plot(data_frames['DEPIN_']['Date_Time'], data_frames['DEPIN_']['Depth'], '.')
        ax.set_xlabel('Time')
        ax.set_ylabel('Depth (m)')
        ax.set_title('DEPIN Data')
        ax.set_xlim(Start, End)
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_DEPIN.png'])

    # Plotting GPS Data
    if 'GPSIN_' in data_frames:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(data_frames['GPSIN_']['Longitude'], data_frames['GPSIN_']['Latitude'], '.')
        ax.set_xlabel('Time')
        ax.set_ylabel('Coordinates')
        ax.set_title('GPS Data over Time')
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_GPSIN.png'])
    # Plotting Position Data
    if 'POSITI' in data_frames:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(data_frames['POSITI']['Longitude'], data_frames['POSITI']['Latitude'], 'k.')
        ax.set_xlabel('Time')
        ax.set_ylabel('Coordinates')
        ax.set_title('POSITI Data')
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_POSITI.png'])
    # Plotting Heave Data
    if 'HEAVE_' in data_frames:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(data_frames['HEAVE_']['Date_Time'], data_frames['HEAVE_']['Surge'], '.k', label='Surge')
        ax.plot(data_frames['HEAVE_']['Date_Time'], data_frames['HEAVE_']['Sway'],  '.b',label='Sway')
        ax.plot(data_frames['HEAVE_']['Date_Time'], data_frames['HEAVE_']['Heave'],  '.g',label='Heave')
        ax.set_xlabel('Time')
        ax.set_ylabel('Heave (m)')
        ax.set_title('HEAVE Data')
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_HEAVE.png'])
    # Plotting Speed Data
    if 'SPEED_' in data_frames:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(data_frames['SPEED_']['Date_Time'], data_frames['SPEED_']['EastSpeed'],  '.',label='East Speed', color='g')
        ax.plot(data_frames['SPEED_']['Date_Time'], data_frames['SPEED_']['NorthSpeed'],  '.',label='North Speed', color='k')
        ax.plot(data_frames['SPEED_']['Date_Time'], data_frames['SPEED_']['UpSpeed'],  '.',label='Up Speed', color='b')
        ax.plot(data_frames['SPEED_']['Date_Time'], data_frames['SPEED_']['Velocity'],  '.',label='Velocity', color='m')
        ax.set_xlabel('Time')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('SPEED Data')
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_SPEED.png'])
    # Plotting STDPOS Data
    if 'STDPOS' in data_frames:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(data_frames['STDPOS']['Date_Time'], data_frames['STDPOS']['Latitude'],  '.',label='Latitude', color='k')
        ax.plot(data_frames['STDPOS']['Date_Time'], data_frames['STDPOS']['Longitude'],  '.',label='Longitude', color='b')
        ax.set_xlabel('Time')
        ax.set_ylabel('Coordinates')
        ax.set_title('STDPOS Data')
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_STDPOS.png'])
    # Plotting Heading Data
    if 'HETHS' in data_frames and 'HEHDT' in data_frames:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(data_frames['HETHS']['Date_Time'], data_frames['HETHS']['Heading'],  '.',label='Heading', color='k')
        ax.plot(data_frames['HEHDT']['Date_Time'], data_frames['HEHDT']['Heading'],  '.',label='Heading', color='b')
        ax.set_xlabel('Time')
        ax.set_ylabel('Heading (degrees)')
        ax.set_title('Heading HETHS and HEHDT Data')
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_HETHS_HEHDT.png'])
    # Plotting ATITUD Data
    if 'ATITUD' in data_frames:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(data_frames['ATITUD']['Date_Time'], data_frames['ATITUD']['Roll'],  '.',label='Roll', color='k')
        ax.plot(data_frames['ATITUD']['Date_Time'], data_frames['ATITUD']['Pitch'],  '.',label='Pitch', color='b')
        ax.set_xlabel('Time')
        ax.set_ylabel('Roll/Pitch (degrees)')
        ax.set_title('ATITUD Data')
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_ATITUD.png'])
    # Plotting STDHRP Data
    if 'STDHRP' in data_frames:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(data_frames['STDHRP']['Date_Time'], data_frames['STDHRP']['HeadingSTD'], '.', label='Heading STD', color='k')
        ax.plot(data_frames['STDHRP']['Date_Time'], data_frames['STDHRP']['RollSTD'],  '.', label='Roll STD', color='b')
        ax.plot(data_frames['STDHRP']['Date_Time'], data_frames['STDHRP']['PitchSTD'],  '.', label='Pitch STD', color='g')
        ax.set_xlabel('Time')
        ax.set_ylabel('Heading/Roll/Pitch STD (degrees)')
        ax.set_title('STDHRP Data')
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_STDHRP.png'])
    # Plotting LOGDVL Data
    if 'LOGDVL' in data_frames:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(data_frames['LOGDVL']['Date_Time'], data_frames['LOGDVL']['DVL_Distance_2btm'],  '.', label='DVL Distance to Bottom', color='k')
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance (m)')
        ax.set_title('LOGDVL Data')
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_LOGDVL.png'])
    # Plotting UTMWGS Data
    if 'UTMWGS' in data_frames:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(data_frames['UTMWGS']['Northing'], data_frames['UTMWGS']['Easting'],  '.', label='Easting', color='k')
        ax.set_xlabel('Coordinates (m)')
        ax.set_ylabel('Coordinates (m)')
        ax.set_title('UTMWGS Data')
        ax.legend()
        if Savefile:
            plt.savefig([Savefile + '_UTMWGS.png'])

def Var_plot(Dir,Filename, Variable, StartDive, EndDive):
    """
    Plots the specified variable from the PHINS data files for a specific dive range.
    
    Args:
        Filename (str): Directory containing the PHINS data files.
        Variable (str): The variable to plot (e.g., 'DEPTH', 'GPSIN_', etc.).
        StartDive (int): Start dive number.
        EndDive (int): End dive number.
    """
    VehicleOutputDir=Dir.replace('Vehicle_Data','Vehicle_Output')
    print(f"VehicleOutputDir: {VehicleOutputDir}")
    df=pd.read_csv(Dir+"\\"+Filename)
    print(f"Reading {Variable} data from {Dir}\\{Filename}")
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df['Date_Time'] = df['Date_Time'].dt.tz_localize('UTC')
    df = df.dropna(subset=['Date_Time'])
    plt.figure(figsize=(19.2, 10.8))
    plt.plot(df['Date_Time'], df[Variable], '.')
    plt.xlabel('Time')
    plt.ylabel(Variable)
    plt.title(f'{Variable} Data')
    plt.xlim(StartDive, EndDive)
    plt.grid()
    plt.tight_layout()
    plt.savefig(VehicleOutputDir + "\\" + f"{Variable}_Phins_Data.png")
    print(f"Saved {Variable} data plot to {VehicleOutputDir}\\{Variable}_Phins_Data.png")
    plt.close()

def basic_analysis(Dir, Avg_dive_time):
    """
    Plots the PHINS data for a specific dive range.
    
    Args:
        VehicleDir (str): Directory containing the PHINS data files.
        VehicleOutputDir (str): Directory to save the output plots.
        StartDive (int): Start dive number.
        EndDive (int): End dive number.
    """
    VehicleDir=os.path.join(Dir, 'Vehicle_Data')
    VehicleOutputDir=os.path.join(Dir, 'Vehicle_Output')
    print(f"VehicleOutputDir: {VehicleOutputDir}")
    # Check if the PHINS data files are available
    phins_check(VehicleDir)
    print(f"PHINS data files checked in {VehicleDir}")
    Avg_dive_time = pd.Timestamp(Avg_dive_time).tz_localize('UTC')
    print(f"Avg_dive_time: {Avg_dive_time}")
    StartDive, EndDive, GPS_OffsetG, GPS_OffsetP = NAV_surface_offset(VehicleDir, VehicleOutputDir, Avg_dive_time)
    print(f"StartDive: {StartDive}, EndDive: {EndDive}, GPS_OffsetG: {GPS_OffsetG}, GPS_OffsetP: {GPS_OffsetP}")
    Var_plot(VehicleDir,"ATITUD.csv", 'Roll', StartDive, EndDive)
    Var_plot(VehicleDir,"ATITUD.csv", 'Pitch', StartDive, EndDive)
    Var_plot(VehicleDir,"DEPIN_.csv", 'Depth', StartDive, EndDive)
    Var_plot(VehicleDir,"HEHDT.csv", 'Heading', StartDive, EndDive)
    Var_plot(VehicleDir,"LOGDVL.csv", 'DVL_Distance_2btm', StartDive, EndDive)
    Var_plot(VehicleDir,"POSITI.csv", 'Latitude', StartDive, EndDive)
    Var_plot(VehicleDir,"POSITI.csv", 'Longitude', StartDive, EndDive)
    Var_plot(VehicleDir,"POSITI.csv", 'Altitude', StartDive, EndDive)
    Var_plot(VehicleDir,"SPEED_.csv", 'EastSpeed', StartDive, EndDive)
    Var_plot(VehicleDir,"SPEED_.csv", 'NorthSpeed', StartDive, EndDive)
    Var_plot(VehicleDir,"SPEED_.csv", 'UpSpeed', StartDive, EndDive)
    Var_plot(VehicleDir,"SPEED_.csv", 'Velocity', StartDive, EndDive)
    Var_plot(VehicleDir,"UTMWGS.csv", 'Easting', StartDive, EndDive)
    Var_plot(VehicleDir,"UTMWGS.csv", 'Northing', StartDive, EndDive)
    Var_plot(VehicleDir,"UTMWGS.csv", 'Altitude', StartDive, EndDive)

    plot_change_rate(VehicleDir, StartDive, EndDive, VehicleOutputDir)

def phins_check(Dir):
    # check if .csv files have been created
    VehicleOutputDir=Dir.replace('Vehicle_Data','Vehicle_Output')

    files_to_check = ['DEPIN_', 'GPSIN_', 'ATITUD', 'UTMWGS', 'LOGDVL', 'HEHDT', 'HETHS']
    for file in files_to_check:
        if not os.path.exists(os.path.join(Dir, file + '.csv')):
            print(f"Phins data not processed yet, running read phins data")
            # find phinsdata_*.bin file in the directory
            file_name = [f for f in os.listdir(Dir) if f.startswith('phinsdata_') and f.endswith('.bin')]
            if not file_name:
                print(f"No phinsdata_*.bin filename could be found in {Dir}")
                return
            file_path = os.path.join(Dir, file_name[0]) 
            data_frames = read_unique_identifiers(file_path)

            # outliers removal
            outliers=outlier_check(data_frames['DEPIN_']['Depth'], data_frames['DEPIN_']['Date_Time'],15, 20, 'Depth',VehicleOutputDir)
            data_frames['DEPIN_'] = data_frames['DEPIN_'][~outliers]

            outliers=outlier_check(data_frames['LOGDVL']['DVL_Distance_2btm'], data_frames['LOGDVL']['Date_Time'],15, 20, 'DVL_Distance_2btm',VehicleOutputDir)
            data_frames['LOGDVL'] = data_frames['LOGDVL'][~outliers]

            for key, df in data_frames.items():
                if not df.empty:
                    output_file = f"{key}.csv"

                    df.to_csv(Dir + "\\" + output_file, index=False)
                    # print(f"Saved {key} data to {output_file}")

def add_IMU_NAV(df, Dir, TimeOffset=0):
    """
    Args:
        df (pd.DataFrame): DataFrame to merge with IMU and navigation data.
            df needs to have a 'Date_Time' column.
        Dir (str): Directory containing the IMU and navigation data files.
    Returns:
        pd.DataFrame: DataFrame with additional columns for IMU and navigation data.
    """

    phins_check(Dir)
    if 'Date_Time' not in df.columns:
        raise ValueError("DataFrame must contain a 'Date_Time' column.")

    df_Atitude=pd.read_csv(Dir+"\\ATITUD.csv")
    # Roll,Pitch,Time_REF,Date_Time
    df_Atitude['Date_Time'] = pd.to_datetime(df_Atitude['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_Atitude['Date_Time'] = df_Atitude['Date_Time'].dt.tz_localize('UTC')
    df_Atitude['Date_Time'] = df_Atitude['Date_Time'] + pd.Timedelta(seconds=TimeOffset)
    df_Atitude = df_Atitude.dropna(subset=['Date_Time'])

    df_HEHDT=pd.read_csv(Dir+"\\HEHDT.csv")
    # Heading,Time_REF,Date_Time
    df_HEHDT['Date_Time'] = pd.to_datetime(df_HEHDT['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_HEHDT['Date_Time'] = df_HEHDT['Date_Time'].dt.tz_localize('UTC')
    df_HEHDT['Date_Time'] = df_HEHDT['Date_Time'] + pd.Timedelta(seconds=TimeOffset)
    df_HEHDT = df_HEHDT.dropna(subset=['Date_Time'])

    df_UTMWGS=pd.read_csv(Dir+"\\UTMWGS.csv")
    # LatZone,LonZone,Easting,Northing,Altitude,Time_REF,Date_Time
    df_UTMWGS['Date_Time'] = pd.to_datetime(df_UTMWGS['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_UTMWGS['Date_Time'] = df_UTMWGS['Date_Time'].dt.tz_localize('UTC')
    df_UTMWGS['Date_Time'] = df_UTMWGS['Date_Time'] + pd.Timedelta(seconds=TimeOffset)
    df_UTMWGS = df_UTMWGS.dropna(subset=['Date_Time'])

    df_DEPIN = pd.read_csv(Dir+"\\DEPIN_.csv")
    # Depth,Time,Time_REF,Date_Time
    df_DEPIN['Date_Time'] = pd.to_datetime(df_DEPIN['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_DEPIN['Date_Time'] = df_DEPIN['Date_Time'].dt.tz_localize('UTC')
    df_DEPIN['Date_Time'] = df_DEPIN['Date_Time'] + pd.Timedelta(seconds=TimeOffset)
    df_DEPIN = df_DEPIN.dropna(subset=['Date_Time'])

    df_LOGDVL = pd.read_csv(Dir+"\\LOGDVL.csv")
    # SoundVelocity,Compensation,DVL_Distance_2btm,Time_REF,Date_Time
    df_LOGDVL['Date_Time'] = pd.to_datetime(df_LOGDVL['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_LOGDVL['Date_Time'] = df_LOGDVL['Date_Time'].dt.tz_localize('UTC')
    df_LOGDVL['Date_Time'] = df_LOGDVL['Date_Time'] + pd.Timedelta(seconds=TimeOffset)
    df_LOGDVL = df_LOGDVL.dropna(subset=['Date_Time'])

    df_SPEED_ = pd.read_csv(Dir+"\\SPEED_.csv")
    # EastSpeed,NorthSpeed,UpSpeed,Velocity,Time_REF,Date_Time
    df_SPEED_['Date_Time'] = pd.to_datetime(df_SPEED_['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_SPEED_['Date_Time'] = df_SPEED_['Date_Time'].dt.tz_localize('UTC')
    df_SPEED_['Date_Time'] = df_SPEED_['Date_Time'] + pd.Timedelta(seconds=TimeOffset)
    df_SPEED_ = df_SPEED_.dropna(subset=['Date_Time'])

    df_POSITI = pd.read_csv(Dir+"\\POSITI.csv")
    # Latitude,Longitude,Altitude,Time_REF,Date_Time
    df_POSITI['Date_Time'] = pd.to_datetime(df_POSITI['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_POSITI['Date_Time'] = df_POSITI['Date_Time'].dt.tz_localize('UTC')
    df_POSITI['Date_Time'] = df_POSITI['Date_Time'] + pd.Timedelta(seconds=TimeOffset)
    df_POSITI = df_POSITI.dropna(subset=['Date_Time'])

    df['Timestamp'] = df['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_Atitude['Timestamp'] = df_Atitude['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_HEHDT['Timestamp'] = df_HEHDT['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_UTMWGS['Timestamp'] = df_UTMWGS['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_DEPIN['Timestamp'] = df_DEPIN['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_LOGDVL['Timestamp'] = df_LOGDVL['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_SPEED_['Timestamp'] = df_SPEED_['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_POSITI['Timestamp'] = df_POSITI['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )

    df_Atitude = df_Atitude.dropna(subset=['Timestamp', 'Roll'])
    df_Atitude = df_Atitude.sort_values('Timestamp')
    df_HEHDT = df_HEHDT.dropna(subset=['Timestamp', 'Heading'])
    df_HEHDT = df_HEHDT.sort_values('Timestamp')
    df_UTMWGS = df_UTMWGS.dropna(subset=['Timestamp', 'Easting', 'Northing'])
    df_UTMWGS = df_UTMWGS.sort_values('Timestamp')
    df_DEPIN = df_DEPIN.dropna(subset=['Timestamp', 'Depth'])
    df_DEPIN = df_DEPIN.sort_values('Timestamp')
    df_LOGDVL = df_LOGDVL.dropna(subset=['Timestamp', 'DVL_Distance_2btm'])
    df_LOGDVL = df_LOGDVL.sort_values('Timestamp')
    df_SPEED_ = df_SPEED_.dropna(subset=['Timestamp', 'Velocity'])
    df_SPEED_ = df_SPEED_.sort_values('Timestamp')
    df_POSITI = df_POSITI.dropna(subset=['Timestamp', 'Latitude', 'Longitude'])
    df_POSITI = df_POSITI.sort_values('Timestamp')
    

    df['AUV_Roll'] = np.interp(df['Timestamp'], df_Atitude['Timestamp'], df_Atitude['Roll'])
    df['AUV_Pitch'] = np.interp(df['Timestamp'], df_Atitude['Timestamp'], df_Atitude['Pitch'])
    df['AUV_Heading'] = np.interp(df['Timestamp'], df_HEHDT['Timestamp'], df_HEHDT['Heading'])
    df['AUV_Northing'] = np.interp(df['Timestamp'], df_UTMWGS['Timestamp'], df_UTMWGS['Northing'])
    df['AUV_Easting'] = np.interp(df['Timestamp'], df_UTMWGS['Timestamp'], df_UTMWGS['Easting'])
    df['AUV_Depth'] = -np.interp(df['Timestamp'], df_DEPIN['Timestamp'], df_DEPIN['Depth'])
    df['AUV_Altitude'] = np.interp(df['Timestamp'], df_LOGDVL['Timestamp'], df_LOGDVL['DVL_Distance_2btm'])
    df['AUV_WaterDepth'] = df['AUV_Depth'] - df['AUV_Altitude'] # depth negative, altitude positive, water depth is negative
    df['AUV_Velocity'] = np.interp(df['Timestamp'], df_SPEED_['Timestamp'], df_SPEED_['Velocity'])
    df['AUV_Latitude'] = np.interp(df['Timestamp'], df_POSITI['Timestamp'], df_POSITI['Latitude'])
    df['AUV_Longitude'] = np.interp(df['Timestamp'], df_POSITI['Timestamp'], df_POSITI['Longitude'])

    return df

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth using the Haversine formula.
    Parameters:
        lat1, lon1: Latitude and Longitude of point 1 in decimal degrees.
        lat2, lon2: Latitude and Longitude of point 2 in decimal degrees.
    Returns:
        Distance in meters.
    """
    # Earth radius in meters
    R = 6371000  

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance
    return R * c

def NAV_surface_offset(Dir, plot_dir, dive_time):
    """
    Calculate the NAV surface offset and plot the results.
    :param data_PHINS: Dictionary of DataFrames containing Phins data.
    :param Dir: Directory to save the plots.
    :param dive_time: mid point of the dive used to find the start and end of the dive.
    """
    phins_check(Dir)

    df_DEPIN = pd.read_csv(Dir+"\\DEPIN_.csv")
    # Depth,Time,Time_REF,Date_Time
    df_DEPIN['Date_Time'] = pd.to_datetime(df_DEPIN['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_DEPIN['Date_Time'] = df_DEPIN['Date_Time'].dt.tz_localize('UTC')

    df_GPSIN_ = pd.read_csv(Dir+"\\GPSIN_.csv")
    # Latitude,Longitude,Altitude,Time,Quality,Time_REF,Date_Time   
    df_GPSIN_['Date_Time'] = pd.to_datetime(df_GPSIN_['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_GPSIN_['Date_Time'] = df_GPSIN_['Date_Time'].dt.tz_localize('UTC')

    df_LOGDVL = pd.read_csv(Dir+"\\LOGDVL.csv")
    # SoundVelocity,Compensation,DVL_Distance_2btm,Time_REF,Date_Time
    df_LOGDVL['Date_Time'] = pd.to_datetime(df_LOGDVL['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_LOGDVL['Date_Time'] = df_LOGDVL['Date_Time'].dt.tz_localize('UTC')

    df_POSITI = pd.read_csv(Dir+"\\POSITI.csv")
    # Latitude,Longitude,Altitude,Time_REF,Date_Time
    df_POSITI['Date_Time'] = pd.to_datetime(df_POSITI['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_POSITI['Date_Time'] = df_POSITI['Date_Time'].dt.tz_localize('UTC')
    
    df_DEPIN['Timestamp'] = df_DEPIN['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_LOGDVL['Timestamp'] = df_LOGDVL['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_POSITI['Timestamp'] = df_POSITI['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_GPSIN_['Timestamp'] = df_GPSIN_['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df= df_DEPIN
    df['Depth'] = -df['Depth']  # depth negative
    df['Latitude'] = np.interp(df['Timestamp'], df_POSITI['Timestamp'], df_POSITI['Latitude'])
    df['Longitude'] = np.interp(df['Timestamp'], df_POSITI['Timestamp'], df_POSITI['Longitude'])
    df['Altitude'] = np.interp(df['Timestamp'], df_LOGDVL['Timestamp'], df_LOGDVL['DVL_Distance_2btm'])
    df['WaterDepth'] = df['Depth'] - df['Altitude'] # depth negative, altitude positive, water depth is negative
    df['GPS_Quality'] = np.interp(df['Timestamp'], df_GPSIN_['Timestamp'], df_GPSIN_['Quality'])
    df['Latitude_GPS'] = np.interp(df['Timestamp'], df_GPSIN_['Timestamp'], df_GPSIN_['Latitude'])
    df['Longitude_GPS'] = np.interp(df['Timestamp'], df_GPSIN_['Timestamp'], df_GPSIN_['Longitude'])


    SurfaceTimes = df[df['Depth'] > -1]
    EndDive = SurfaceTimes[SurfaceTimes['Date_Time'] > dive_time]['Date_Time'].min()
    StartDive= SurfaceTimes[SurfaceTimes['Date_Time'] < dive_time]['Date_Time'].max()

    df['Distance_m'] = haversine(
        df['Latitude'], df['Longitude'],
        df['Latitude_GPS'], df['Longitude_GPS']
    )

    df['TimeSinceSurface'] = (df['Date_Time'] - EndDive).dt.total_seconds() / 60
    GPS1 = df[df['GPS_Quality'] == 1]
    GPS2 = df[df['GPS_Quality'] == 2]
    Good_GPS_Data = df[(df['GPS_Quality'] == 2) & (df['TimeSinceSurface'] < 5)  & (df['TimeSinceSurface'] > 0)]
    Poor_GPS_Data = df[(df['GPS_Quality'] == 1) & (df['TimeSinceSurface'] < 5) & (df['TimeSinceSurface'] > 0)]
    plt.figure(figsize=(19.2, 10.8))
    plt.subplot(2, 1, 1)
    plt.plot(GPS1['TimeSinceSurface'], GPS1['Distance_m'], 'r.', label='GPS Quality 1')
    plt.plot(GPS2['TimeSinceSurface'], GPS2['Distance_m'], 'g.', label='GPS Quality 2')
    plt.legend()
    plt.xlim(0, 30)
    plt.ylim(0, 25)
    plt.xlabel('Time Since Surface (min)')
    plt.ylabel('Distance (m)')
    plt.title('Phins NAV Position to GPS Post Dive')
    plt.subplot(2, 1, 2)
    plt.plot(df['TimeSinceSurface'], df['Altitude'], 'b.', label='DVL Distance to Bottom')
    plt.xlim(0, 30)
    plt.xlabel('Time Since Surface (min)')
    plt.ylabel('DVL Distance to Bottom (m)')
    plt.savefig(os.path.join(plot_dir, 'IMU_Position_to_GPS_Post_Dive.png'))
    plt.close()

    # remove outliers Distance_m
    GPS1 = GPS1[np.abs(GPS1['Distance_m'] - GPS1['Distance_m'].mean()) < 3 * GPS1['Distance_m'].std()]
    GPS2 = GPS2[np.abs(GPS2['Distance_m'] - GPS2['Distance_m'].mean()) < 3 * GPS2['Distance_m'].std()]
    
    GPS1 = GPS1[(GPS1['TimeSinceSurface'] < 15) & (GPS1['TimeSinceSurface'] > 0)]
    GPS2 = GPS2[(GPS2['TimeSinceSurface'] < 15) & (GPS2['TimeSinceSurface'] > 0)]

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(GPS1['TimeSinceSurface'], GPS1['Distance_m'], 'r.', label='GPS Quality 1')
    plt.plot(GPS2['TimeSinceSurface'], GPS2['Distance_m'], 'b.', label='GPS Quality 2')
    plt.legend()
    plt.xlim(0, 15)
    # plt.ylim(0, 25)
    plt.xlabel('Time Since Surface (min)')
    plt.ylabel('Distance (m)')
    plt.savefig(os.path.join(plot_dir, 'Report_plot_IMU_Position_to_GPS_Post_Dive.png'))
    plt.close()

    GPS_OffsetG = Good_GPS_Data['Distance_m'].mean() if not Good_GPS_Data.empty else 0
    GPS_OffsetP = Poor_GPS_Data['Distance_m'].mean() if not Poor_GPS_Data.empty else 0

    return StartDive, EndDive, GPS_OffsetG, GPS_OffsetP

def plot_change_rate(Dir, Start, End, plot_dir):
    """
    Plot the change rate of the data in the given directory.
    Foucs on the change rate of the depth, speed, and position data and couple it with the IMU data
    Args:
        Dir (str): Directory containing the data files.
        Start (str): Start time in 'YYYY-MM-DD HH:MM:SS' format.
        End (str): End time in 'YYYY-MM-DD HH:MM:SS' format.
        Savefile (str, optional): If provided, save the plot to this file.
    """
    phins_check(Dir)

    df_Atitude=pd.read_csv(Dir+"\\ATITUD.csv")
    # Roll,Pitch,Time_REF,Date_Time
    df_Atitude['Date_Time'] = pd.to_datetime(df_Atitude['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_Atitude['Date_Time'] = df_Atitude['Date_Time'].dt.tz_localize('UTC')

    df_HEHDT=pd.read_csv(Dir+"\\HEHDT.csv")
    # Heading,Time_REF,Date_Time
    df_HEHDT['Date_Time'] = pd.to_datetime(df_HEHDT['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_HEHDT['Date_Time'] = df_HEHDT['Date_Time'].dt.tz_localize('UTC')

    df_UTMWGS=pd.read_csv(Dir+"\\UTMWGS.csv")
    # LatZone,LonZone,Easting,Northing,Altitude,Time_REF,Date_Time
    df_UTMWGS['Date_Time'] = pd.to_datetime(df_UTMWGS['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_UTMWGS['Date_Time'] = df_UTMWGS['Date_Time'].dt.tz_localize('UTC')

    df_DEPIN = pd.read_csv(Dir+"\\DEPIN_.csv")
    # Depth,Time,Time_REF,Date_Time
    df_DEPIN['Date_Time'] = pd.to_datetime(df_DEPIN['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_DEPIN['Date_Time'] = df_DEPIN['Date_Time'].dt.tz_localize('UTC')

    df_LOGDVL = pd.read_csv(Dir+"\\LOGDVL.csv")
    # SoundVelocity,Compensation,DVL_Distance_2btm,Time_REF,Date_Time
    df_LOGDVL['Date_Time'] = pd.to_datetime(df_LOGDVL['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_LOGDVL['Date_Time'] = df_LOGDVL['Date_Time'].dt.tz_localize('UTC')

    df_SPEED_ = pd.read_csv(Dir+"\\SPEED_.csv")
    # EastSpeed,NorthSpeed,UpSpeed,Velocity,Time_REF,Date_Time
    df_SPEED_['Date_Time'] = pd.to_datetime(df_SPEED_['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_SPEED_['Date_Time'] = df_SPEED_['Date_Time'].dt.tz_localize('UTC')

    df_POSITI = pd.read_csv(Dir+"\\POSITI.csv")
    # Latitude,Longitude,Altitude,Time_REF,Date_Time
    df_POSITI['Date_Time'] = pd.to_datetime(df_POSITI['Date_Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df_POSITI['Date_Time'] = df_POSITI['Date_Time'].dt.tz_localize('UTC')


    df_DEPIN['Timestamp'] = df_DEPIN['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_LOGDVL['Timestamp'] = df_LOGDVL['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_SPEED_['Timestamp'] = df_SPEED_['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_Atitude['Timestamp'] = df_Atitude['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df_POSITI['Timestamp'] = df_POSITI['Date_Time'].apply(
        lambda x: x.timestamp() if pd.notnull(x) else None
    )
    df= df_DEPIN
    df['Depth'] = -df['Depth']  # depth negative
    df['Pitch'] = np.interp(df['Timestamp'], df_Atitude['Timestamp'], df_Atitude['Pitch'])
    df['UpSpeed'] = np.interp(df['Timestamp'], df_SPEED_['Timestamp'], df_SPEED_['UpSpeed'])
    df['EastSpeed'] = np.interp(df['Timestamp'], df_SPEED_['Timestamp'], df_SPEED_['EastSpeed'])
    df['NorthSpeed'] = np.interp(df['Timestamp'], df_SPEED_['Timestamp'], df_SPEED_['NorthSpeed'])
    df['Altitude'] = np.interp(df['Timestamp'], df_LOGDVL['Timestamp'], df_LOGDVL['DVL_Distance_2btm'])
    df['Latitude'] = np.interp(df['Timestamp'], df_POSITI['Timestamp'], df_POSITI['Latitude'])
    df['Longitude'] = np.interp(df['Timestamp'], df_POSITI['Timestamp'], df_POSITI['Longitude'])

    # Remove data outside the specified time range
    df= df[(df['Date_Time'] >= pd.to_datetime(Start)) & (df['Date_Time'] <= pd.to_datetime(End))]

    df=df.dropna(subset=['Date_Time'])
    # calc differnece between consecutive values and remove outliers
    df['DepthRateChange'] = df['Depth'].diff().fillna(0)/ (df['Date_Time'].diff().dt.total_seconds().fillna(1))
    df['Depth_Change_'] = df['Depth'].diff().fillna(0)
    df['Depth_Change'] = df['Depth_Change_'].abs()
    df['UpSpeed_Change'] = df['UpSpeed'].diff().fillna(0)
    df['UpSpeed_Change'] = df['UpSpeed_Change'].abs()
    df['Pitch_Change'] = df['Pitch'].diff().fillna(0)
    df['Pitch_Change'] = df['Pitch_Change'].abs()

    # devation from rolling mean
    TimeChange = df['Date_Time'].diff().dt.total_seconds().fillna(1).mean()
    SmoothWindow = int(60 / TimeChange)  # 60 seconds smoothing window
    df['AltitudeDevationFromMean']= df['Altitude'] - df['Altitude'].rolling(window=SmoothWindow, min_periods=1, center=True).mean()

    # save csv file with Lat, Lon, AltitudeDevationFromMean
    df[['Latitude', 'Longitude', 'AltitudeDevationFromMean']].to_csv(os.path.join(plot_dir, 'AltitudeDevationFromMean.csv'), index=False)

    plt.figure(figsize=(19.2, 10.8))
    plt.subplot(2, 1, 1)
    plt.plot(df['Date_Time'], df['Depth'], 'b.', label='Depth')
    plt.plot(df['Date_Time'], df['Altitude'], 'g.', label='Altitude')
    plt.plot(df['Date_Time'], df['Depth']-df['Altitude'], 'r.', label='Water Depth')
    plt.legend()
    plt.xlim(pd.to_datetime(Start), pd.to_datetime(End))
    plt.ylabel('Depth (m)')
    plt.subplot(2, 1, 2)
    plt.plot(df['Date_Time'], df['AltitudeDevationFromMean'], 'b.', label='Depth Rate Change')
    plt.xlim(pd.to_datetime(Start), pd.to_datetime(End))
    plt.ylabel('Altitude Deviation from Mean (m)')
    plt.xlabel('Time')    
    plt.savefig(os.path.join(plot_dir, 'AltitudeDevationFromMean.png'))
    plt.close()

    # remove outliers hard code max rate change
    df=df[df['Depth_Change'] < 5]
    avg_sample_rate = df['Date_Time'].diff().dt.total_seconds().mean()
    print(f"Average sample rate: {avg_sample_rate} seconds")
    df['Velocity'] = np.sqrt(df['EastSpeed']**2 + df['NorthSpeed']**2 + df['UpSpeed']**2)
    avg_velocity = df['Velocity'].mean()
    print(f"Average velocity: {avg_velocity} m/s")

    # remove outliers hard code max rate change
    df=df[df['DepthRateChange'].abs() < 2/avg_sample_rate]  # 2 m/s is a reasonable max rate change for AUVs
    
    df['SlopeResponse']= df['DepthRateChange'] / df['Velocity'].replace(0, np.nan)  # avoid division by zero
    df['DepthRateChange_Smooth'] = df['DepthRateChange'].rolling(window=50, min_periods=1, center=True).mean()
    Avg_altitude = df['Altitude'].mean()
    first_altitude_below_avg = df[df['Altitude'] < Avg_altitude].index.min()
    print(f"First Altitude below average: {first_altitude_below_avg}")
    last_altitude_below_avg = df[df['Altitude'] < Avg_altitude].index.max()
    print(f"Last Altitude below average: {last_altitude_below_avg}")
    Dive_Section = df.loc[:first_altitude_below_avg]
    Surface_Section = df.loc[last_altitude_below_avg:]
    Survey_Section = df.loc[first_altitude_below_avg:last_altitude_below_avg]

    # alt how hard is it trying to correct, how long does it take to correct
    # differntial of altitutde shows areas of concern
    df['AltitudeRateChange'] = df['Altitude'].diff().fillna(0) / (df['Date_Time'].diff().dt.total_seconds().fillna(1))
    df['AltitudeRateChange_Smooth'] = df['AltitudeRateChange'].rolling(window=10, min_periods=1, center=True).mean()
    df['Pitch_Smooth'] = df['Pitch'].rolling(window=10, min_periods=1, center=True).mean()
    Dive_Section = df.loc[:first_altitude_below_avg]
    Surface_Section = df.loc[last_altitude_below_avg:]
    Survey_Section = df.loc[first_altitude_below_avg:last_altitude_below_avg]
    best_lag, time_offset_sec, corr = find_best_time_offset(Survey_Section, col1='DepthRateChange_Smooth', col2='Pitch_Smooth')
    print("avg Altitude: ", Survey_Section['Altitude'].mean())

    # plt.figure(figsize=(19.2, 10.8))
    # plt.subplot(3, 1, 1)
    # plt.plot(df['Date_Time'], -df['AltitudeRateChange_Smooth'], 'k', label='Altitude')
    # plt.plot(df['Date_Time'], df['DepthRateChange_Smooth'], 'b.', label='Veh Depth')
    # plt.plot(df['Date_Time'], df['AltitudeRateChange']+ df['DepthRateChange'], 'r', label='Veh Depth + Altitude')
    # plt.ylabel('Rate Change (m/s)')
    # plt.subplot(3, 1, 2)
    # plt.plot(df['Date_Time'], df['Depth'], 'r.', label='SURVEY')
    # plt.plot(df['Date_Time'], df['Altitude'], 'b.', label='WATER DEPTH')
    # plt.plot(df['Date_Time'], df['Depth']-df['Altitude'], 'g.', label='WATER DEPTH')
    # plt.subplot(3, 1, 3)
    # plt.plot(df['Date_Time'], df['Pitch'], 'k.', label='DEPTH')
    # plt.show()

    # plt.figure(figsize=(19.2, 10.8))
    # plt.plot(df['Date_Time'], df['AltitudeRateChange_Smooth'], 'k', label='Altitude')
    # plt.plot(df['Date_Time'], df['Pitch_Smooth']*.1, 'b', label='Pitch')
    # plt.plot(df['Date_Time'], df['DepthRateChange_Smooth'], 'r', label='Veh Depth')
    # plt.grid()
    # plt.xlabel('Time')
    # plt.show()

    # VERTICAL CHANGE RATE
    plt.figure(figsize=(19.2, 10.8))
    plt.subplot(3, 1, 1)
    plt.plot(Dive_Section['Date_Time'], Dive_Section['Depth'], 'k.', label='DEPTH')
    plt.plot(Surface_Section['Date_Time'], Surface_Section['Depth'], 'g.', label='SURFACE')
    plt.plot(Survey_Section['Date_Time'], Survey_Section['Depth'], 'r.', label='SURVEY')
    plt.plot(df['Date_Time'], df['Depth']-df['Altitude'], 'b.', label='WATER DEPTH')
    # plt.xlim(pd.to_datetime(Start), pd.to_datetime(End))
    plt.ylabel('Depth (m)')
    plt.subplot(3, 1, 2)
    plt.plot(df['Date_Time'], df['DepthRateChange'], 'b.', label='WATER DEPTH')
    # plt.plot(df['Date_Time'], df['UpSpeed'], 'b.', label='UP SPEED')
    plt.xlabel('Time')
    plt.ylabel('Vertical Speed (m/s)')
    plt.xlim(pd.to_datetime(Start), pd.to_datetime(End))
    plt.subplot(3, 1, 3)
    plt.plot(df['Date_Time'], df['Pitch'], 'r.', label='PITCH')
    plt.xlabel('Time')  
    plt.ylabel('Change Rate')
    plt.title('Pitch')
    plt.xlim(pd.to_datetime(Start), pd.to_datetime(End))
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'Vertical_Change_Rate.png'))
    # plt.show()
    plt.close()

    shifted_pitch = Survey_Section['Pitch_Smooth'].shift(best_lag)

    plt.plot(Survey_Section['Pitch_Smooth'].shift(best_lag).iloc[::10], Survey_Section['DepthRateChange_Smooth'].iloc[::10], 'k.', label='SURVEY')
    plt.plot(Dive_Section['Pitch_Smooth'].shift(best_lag).iloc[::10], Dive_Section['DepthRateChange_Smooth'].iloc[::10], 'r.', label='DIVE')
    plt.plot(Surface_Section['Pitch_Smooth'].shift(best_lag).iloc[::10], Surface_Section['DepthRateChange_Smooth'].iloc[::10], 'g.', label='SURFACE')
    plt.xlabel('Pitch (degrees)')  
    plt.ylabel('Up Speed (m/s)')
    plt.title('Pitch vs Up Speed')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, 'Pitch_vs_UpSpeed.png'))
    # plt.show()
    plt.close()

    # plot every 10th value
    plt.figure(figsize=(19.2, 10.8))
    plt.plot(df['DepthRateChange_Smooth'].iloc[::10], df['UpSpeed'].iloc[::10], 'b.', label='Depth Change vs Up Speed')
    plt.xlabel('Depth Change (m)')
    plt.ylabel('Up Speed (m/s)')
    plt.title('Depth Change vs Up Speed')
    plt.grid()
    # plt.xlim(-5, 5)
    # plt.ylim(-2, 2)
    plt.savefig(os.path.join(plot_dir, 'Depth_Change_vs_UpSpeed.png'))
    # plt.show()
    plt.close()


    
import numpy as np
def find_best_time_offset(df, col1='AltitudeRateChange_Smooth', col2='Pitch_Smooth'):
    # Ensure no NaNs and both columns are aligned
    valid = df[[col1, col2]].dropna()
    x = valid[col1].values
    y = valid[col2].values

    # Subtract mean to remove DC component
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Compute cross-correlation
    corr = np.correlate(x, y, mode='full')
    lags = np.arange(-len(x) + 1, len(x))

    # Find the lag with the maximum absolute correlation
    best_lag = lags[np.argmax(np.abs(corr))]

    # Calculate time offset in seconds (assuming regular sampling)
    sample_interval = np.median(np.diff(df['Date_Time'].dropna().astype(np.int64) / 1e9))
    time_offset_sec = best_lag * sample_interval

    print(f"Best lag: {best_lag} samples, Time offset: {time_offset_sec:.2f} seconds")
    return best_lag, time_offset_sec, corr

def outlier_check( Variable, Time,max_change_rate, varince_max, variable_name, plot_dir):
    """
    Check for outliers in the given variable based on the maximum change rate.
    Args:
        Time (pd.Series): Time series data.
        Variable (pd.Series): Variable to check for outliers.
        max_change_rate (float): Maximum allowed change rate.
        plot_dir (str): Directory to save the plot.
    Returns:
        pd.Series: Boolean series indicating outliers.
    """
    # outlier based on max change rate
    time_diff_sec = Time.diff().dt.total_seconds().abs()
    change_rate = Variable.diff().abs() / time_diff_sec
    outliers = change_rate > max_change_rate

    # outlier with respect to the running mean and std deviation
    rolling_mean = Variable.rolling(window=50, center=True).mean()
    variance_from_mean = Variable - rolling_mean
    variance_std = variance_from_mean.std()
    outliers |= (variance_from_mean.abs() > varince_max)
    outliers2 = (variance_from_mean.abs() > varince_max)

    # plt.figure(figsize=(19.2, 10.8))
    # plt.subplot(2, 1, 1)
    # plt.plot(Time, change_rate, label='Variable', color='b')
    # plt.axhline(max_change_rate, color='r', linestyle='--', label='Max Change Rate')
    # plt.xlabel('Time')
    # plt.ylabel('Change Rate')
    # plt.title(f'Change Rate of {variable_name}')
    # plt.subplot(2, 1, 2)
    # plt.plot(Time, Variable, label='Variable', color='b')
    # plt.plot(Time,rolling_mean, label='Rolling Mean', color='orange')
    # plt.plot(Time[outliers], Variable[outliers], 'ro', label='Outliers')
    # plt.plot(Time[outliers2], Variable[outliers2], 'go', label='Outliers (Variance)')
    # plt.xlabel('Time')
    # plt.ylabel(variable_name)
    # plt.title(f'Outlier Check for {variable_name}')
    # plt.show()

    plt.figure(figsize=(19.2, 10.8))
    plt.plot(Time, Variable, label='Variable', color='b')
    plt.plot(Time[outliers], Variable[outliers], 'ro', label='Outliers')
    plt.plot(Time[outliers2], Variable[outliers2], 'go', label='Outliers (Variance)')
    plt.xlabel('Time')
    plt.ylabel('Variable')
    plt.title('Outlier Check')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'outlier_check_{variable_name}.png'))
    plt.close()


    return outliers