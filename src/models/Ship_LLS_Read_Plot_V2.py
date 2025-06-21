import os
import fnmatch
import pandas as pd
import numpy as np
import datetime
import csv
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import glob
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
from models import read_phinsdata as phins

    
def FindSidelobe(Data, Raduis=3.5, MinIntensity=90):
    """
    Find sidelobe points in the data based on intensity and distance.
    :param Data: Input data array.
    :param Raduis: Radius around AUV in meters that is considered sidelobe points.
    :param MinIntensity: Minimum intensity threshold. If high intensity points are near the AUV, assume they are real points.
    :return: Filtered data and number of sidelobe points.
    """
    # Calculate the distance from the center point for X-Z plane
    Dist = np.sqrt((Data[:, 1])**2 + (Data[:, 3])**2)
    # Find points that are within the radius and above the minimum intensity
    index = np.where((Dist > Raduis) | (Data[:, 4] > MinIntensity))[0]
    NumSLpts = len(Data)-len(index)
    return Data[index,:], NumSLpts, index

def CordTransformZ(LS_Z, pitch, roll):
    """
    Transform coordinates based on pitch, roll, heading, and depth.
    :param LS_Z: Input Z coordinates from Voyis refrence frame.
    :param pitch: Pitch angle in degrees.
    :param roll: Roll angle in degrees.
    :return: Transformed Z coordinate. Distance from AUV to seafloor.
    """
    # Convert angles from degrees to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # Apply the transformation
    Z = LS_Z * np.cos(pitch_rad) * np.cos(roll_rad)
    return Z


def LS_First_Look_Plots(dfLS, data_PHINS, Dir, DIVE_NAME):
    """
    Create first look plots for the data.
    :param dfLS: DataFrame containing the processed data.
    :param data_PHINS: Dictionary of DataFrames containing Phins data.
    :param Dir: Directory to save the plots.
    :param DIVE_NAME: Name of the dive.
    """

def LLS_Check(LLSDir, LLSOutputDir, xyz_files, log_callback=None):
    """
    Check the LLS directory and output directory.
    :param LLSDir: Directory containing LLS data.
    :param LLSOutputDir: Directory to save processed LLS data.
    :param xyz_files: List of .xyz files in the LLS directory.
    :param log_callback: Optional callback function for logging messages to GUI
    """
    def log_message(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    if not os.path.exists(LLSDir):
        raise FileNotFoundError(f"LLS directory does not exist: {LLSDir}")
    
    if not os.path.exists(LLSOutputDir):
        os.makedirs(LLSOutputDir)
        print(f"Created output directory: {LLSOutputDir}")
    
    if len(xyz_files) == 0:
        raise FileNotFoundError(f"No .xyz files found in {LLSDir}")
    
    print(f"Found {len(xyz_files)} .xyz files in {LLSDir}")
    log_message(f"Found {len(xyz_files)} .xyz files in {LLSDir}")

def Estiamted_Run_Time(xyz_files, Dir, process_speed=4.53, log_callback=None):
    """
    Estimate the run time based on the size of the .xyz files.
    :param xyz_files: List of .xyz files.
    :param Dir: Directory containing the .xyz files.
    :param log_callback: Optional callback function for logging messages to GUI
    """
    def log_message(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    file_sizes = []
    total_size = 0
    for xyz_file in xyz_files:
        size_bytes = os.path.getsize(xyz_file)
        size_mb = size_bytes / (1024 * 1024)  # Convert to MB
        file_sizes.append(size_mb)
        total_size += size_mb
    
    # estimate run time in minutes based on file size and process speed
    estimated_run_time = total_size / process_speed / 60  
    print(f"Found {len(xyz_files)} .xyz files with total size of {total_size:.2f} MB")
    print(f"Estimated Run Time: {estimated_run_time:.2f} minutes")
    log_message(f"Found {len(xyz_files)} .xyz files with total size of {total_size:.2f} MB")
    log_message(f"Estimated Run Time: {estimated_run_time:.2f} minutes")
    
    start_time = datetime.datetime.now()
    print(f"Estimated finish time: {start_time + datetime.timedelta(minutes=estimated_run_time)}")
    log_message(f"Estimated finish time: {start_time + datetime.timedelta(minutes=estimated_run_time)}")
    return total_size, start_time

def process_xyz_file(file_path, file_number, radius, min_intensity_threshold, bad_point_threshold, vehicle_dir, log_callback=None):   
    def log_message(message):
        print(message)
        if log_callback:
            log_callback(message)

    data = pd.read_csv(file_path, delimiter=',', dtype=float).to_numpy()
    # try:
    #     data = pd.read_csv(file_path, delimiter=',', dtype=float).to_numpy()
    # except ValueError:
    #     df = pd.read_csv(file_path, delimiter=',', low_memory=False)
    #     # Convert all columns to numeric, replacing non-numeric with NaN
    #     df = df.apply(pd.to_numeric, errors='coerce')
    #     data = df.to_numpy()

    unique_times = np.unique(data[:,0])
    time_diff = np.diff(unique_times).mean()
    unique_times_dt = [datetime.datetime.fromtimestamp(t / 1000000, tz=datetime.timezone.utc) 
                      for t in unique_times]
    Valid_Data_Mask = np.zeros((data.shape[0], 2), dtype=int)
    
    # Get IMU data
    df_imu_nav = pd.DataFrame()
    df_imu_nav['Date_Time'] = unique_times_dt
    df_imu_nav = phins.add_IMU_NAV(df_imu_nav, vehicle_dir)

    # Process each timestamp
    rows = []
    for i in range(len(unique_times)):
        time_index = np.where(data[:,0] == unique_times[i])[0]
        row,Valid_Data_Mask = process_timestamp_data(data, time_index, unique_times[i], file_number, 
                                   file_path, df_imu_nav, i, radius, 
                                   min_intensity_threshold, bad_point_threshold,Valid_Data_Mask,time_diff)
        rows.append(row)

    # replace LLS_ with Mask_
    file_path_Mask = file_path.replace('LLS_', 'Mask_')
    np.savetxt(file_path_Mask, Valid_Data_Mask, delimiter=',', fmt='%d')
    print(f"Processed {len(rows)} timestamps from file {file_path}")

    return rows

def calc_point_density(data_good, velocity, time_diff):
    """Calculate point density and add it to the data."""
    # Calculate point density
    if len(data_good) > 0:
        x = data_good[:, 1]
        # get the max distance between X points
        cross_distance= np.max(x)- min(x)
        time_diff = time_diff / 1e6  # Convert microseconds to seconds
        area = cross_distance * time_diff * velocity  # Area in m^2
        area_cm2 = area * 10000  # Convert to cm^2
        point_density = len(data_good) / area_cm2  # Points per square meter
        return point_density

def process_timestamp_data(data, time_index, time_unix, file_number, file_path, 
                         df_imu_nav, i, radius, min_intensity, bad_point_threshold, Valid_Data_Mask,time_diff):
    """Process data for a specific timestamp."""
    date_time = datetime.datetime.fromtimestamp(time_unix/1000000, 
                                              tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')
    total_points = len(time_index)
    data_good, num_sl_pts, data_good_index = FindSidelobe(data[time_index, :], Raduis=radius, MinIntensity=min_intensity)
    good_indices_in_data = time_index[data_good_index]
    Valid_Data_Mask[good_indices_in_data, 0] = 1
    # Initialize metrics
    metrics = initialize_metrics()
    
    if data_good.shape[0] > 100:
        # Calculate intensity metrics
        metrics = calculate_intensity_metrics(data_good, metrics)
        
        # Process intensity thresholds and calculate depths
        metrics = process_depth_data(data_good, df_imu_nav, i, total_points, 
                                   min_intensity, bad_point_threshold, metrics)
    point_density = np.nan
    if metrics['depth_flag'] == 1:
        Valid_Data_Mask[time_index, 1] = 1 # Mark valid depth points in the mask
        point_density=calc_point_density(data_good, df_imu_nav['AUV_Velocity'][i], time_diff)

    # Create the row dictionary
    return create_row_dict(file_number, file_path, date_time, time_unix, 
                         total_points, num_sl_pts, df_imu_nav, i, metrics,point_density), Valid_Data_Mask

def initialize_metrics():
    """Initialize all metrics with default values."""
    return {
        'depth_approach': 0,
        'depth_flag': 0,
        'depth': np.nan,
        'depth25': np.nan,
        'depth75': np.nan,
        'depth_max': np.nan,
        'depth_min': np.nan,
        'depth_std': np.nan,
        'intensity_btm': np.nan,
        'intensity_avg': np.nan,
        'intensity_max': np.nan,
        'intensity_min': np.nan,
        'intensity_std': np.nan,
        'intensity_25th': np.nan,
        'intensity_75th': np.nan,
        'num_li_pts': 0,
        'percent_li_points': 0
    }

def calculate_intensity_metrics(data, metrics):
    """Calculate basic intensity metrics from data."""
    metrics['intensity_avg'] = np.mean(data[:, 4])
    metrics['intensity_max'] = np.max(data[:, 4])
    metrics['intensity_min'] = np.min(data[:, 4])
    metrics['intensity_std'] = np.std(data[:, 4])
    metrics['intensity_25th'] = np.percentile(data[:, 4], 25)
    metrics['intensity_75th'] = np.percentile(data[:, 4], 75)
    return metrics

def process_depth_data(data_good, df_imu_nav, i, total_points, min_intensity, bad_point_threshold, metrics):
    """Process depth data using appropriate approach."""
    index_intensity_thresh = np.where(data_good[:, 4] > min_intensity)[0]
    metrics['num_li_pts'] = len(data_good) - len(index_intensity_thresh)
    metrics['percent_li_points'] = (total_points - len(index_intensity_thresh)) / total_points * 100
    
    if metrics['percent_li_points'] < bad_point_threshold:
        metrics = calculate_depth_approach_1(data_good, index_intensity_thresh, df_imu_nav, i, metrics)
    else:
        metrics = calculate_depth_approach_2(data_good, df_imu_nav, i, metrics)
    
    return metrics

def calculate_depth_approach_1(data_good, index_intensity_thresh, df_imu_nav, i, metrics):
    """Calculate depth using approach 1 (high intensity returns)."""
    metrics['depth_approach'] = 1
    
    # Transform depth values
    depth_values = -CordTransformZ(data_good[index_intensity_thresh, 3], 
                                 df_imu_nav['AUV_Pitch'][i], 
                                 df_imu_nav['AUV_Roll'][i]) + df_imu_nav['AUV_Depth'][i]
    
    # Calculate statistics
    metrics['depth'] = np.mean(depth_values)
    metrics['depth25'] = np.percentile(depth_values, 25)
    metrics['depth75'] = np.percentile(depth_values, 75)
    metrics['depth_max'] = np.max(depth_values)
    metrics['depth_min'] = np.min(depth_values)
    metrics['depth_std'] = np.std(depth_values)
    metrics['intensity_btm'] = np.mean(data_good[index_intensity_thresh, 4])
    
    # Check if depth is valid
    bottom_diff = metrics['depth'] - (df_imu_nav['AUV_Altitude'][i] + df_imu_nav['AUV_Depth'][i])
    bottom_diff25 = metrics['depth25'] - (df_imu_nav['AUV_Altitude'][i] + df_imu_nav['AUV_Depth'][i])
    bottom_diff75 = metrics['depth75'] - (df_imu_nav['AUV_Altitude'][i] + df_imu_nav['AUV_Depth'][i])
    
    if (abs(bottom_diff) < 1.7 or abs(bottom_diff25) < 1.7 or abs(bottom_diff75) < 1.7):
        metrics['depth_flag'] = 1
        
    return metrics

def calculate_depth_approach_2(data_good, df_imu_nav, i, metrics):
    """Calculate depth using approach 2 (density-based)."""
    # Find center points
    index_c = np.where((data_good[:,1] > -0.5) & (data_good[:,1] < 0.5))[0]
    
    if len(index_c) > 10:
        metrics['depth_approach'] = 2
        depth_values = -CordTransformZ(data_good[index_c,3], 
                                     df_imu_nav['AUV_Pitch'][i], 
                                     df_imu_nav['AUV_Roll'][i]) + df_imu_nav['AUV_Depth'][i]
        
        # Use KDE to find peak density
        kde = gaussian_kde(depth_values, bw_method=0.2)
        density = kde(depth_values)
        peak_density_index = np.argmax(density)
        metrics['depth'] = depth_values[peak_density_index]
        metrics['intensity_btm'] = np.mean(data_good[index_c, 4][peak_density_index])
        
        # Check if depth is valid
        bottom_diff = metrics['depth'] - (df_imu_nav['AUV_Altitude'][i] + df_imu_nav['AUV_Depth'][i])
        if abs(bottom_diff) < 1.5:
            metrics['depth_flag'] = 1
            
    return metrics

def create_row_dict(file_number, file_name, date_time, time_unix, 
                  total_points, num_sl_pts, df_imu_nav, i, metrics,point_density):
    """Create a dictionary with all processed data for a row."""
    return {
        'FileNumber': file_number,
        'FileName': file_name,
        'Date_Time': date_time, 
        'TimeUnix': time_unix,
        'TotalPoints': total_points, 
        'NumLIpts': metrics['num_li_pts'],
        'NumSLpts': num_sl_pts,
        'PercentLIPoints': metrics['percent_li_points'],
        'AUV_Heading': df_imu_nav['AUV_Heading'][i],
        'AUV_Pitch': df_imu_nav['AUV_Pitch'][i],
        'AUV_Roll': df_imu_nav['AUV_Roll'][i],
        'AUV_Depth': df_imu_nav['AUV_Depth'][i],
        'AUV_Altitude': df_imu_nav['AUV_Altitude'][i],
        'AUV_Easting': df_imu_nav['AUV_Easting'][i],
        'AUV_Northing': df_imu_nav['AUV_Northing'][i],
        'AUV_Velocity': df_imu_nav['AUV_Velocity'][i],
        'Depth_Approach': metrics['depth_approach'],
        'Depth': metrics['depth'],
        'Depth25': metrics['depth25'],
        'Depth75': metrics['depth75'],
        'DepthMax': metrics['depth_max'],
        'DepthMin': metrics['depth_min'],
        'DepthStd': metrics['depth_std'],
        'AverageIntensityAll': metrics['intensity_avg'],
        'MaxIntensityAll': metrics['intensity_max'],
        'MinIntensityAll': metrics['intensity_min'],
        'StdIntensityAll': metrics['intensity_std'],
        '25thPercentIntensityAll': metrics['intensity_25th'],
        '75thPercentIntensityAll': metrics['intensity_75th'],
        'AverageIntensityBtm': metrics['intensity_btm'],
        'DepthFlag': metrics['depth_flag'],
        'PointDensity': point_density
    }

    
def Summary_plots(dfLS, plot_dir, DIVE_NAME, df_Full_Dive, MIN_INTENSITY_THRESHOLD=None, log_callback=None):
    """
    Create summary plots for the processed data.
    :param dfLS: DataFrame containing the processed data.
    :param plot_dir: Directory to save the plots.
    :param DIVE_NAME: Name of the dive.
    :param df_Full_Dive: Full dive navigation data.
    :param MIN_INTENSITY_THRESHOLD: Minimum intensity threshold used.
    :param log_callback: Optional callback function for logging messages to GUI
    """
    def log_message(message):
        print(message)
        if log_callback:
            log_callback(message)
    
    log_message("Generating dive profile plot...")
    
    dfLS['Distance_Traveled'] = np.sqrt(
        (dfLS['AUV_Easting'].diff())**2 + (dfLS['AUV_Northing'].diff())**2
    ).fillna(0).cumsum()

    dfLS = dfLS[(dfLS['AUV_Easting'] != 0) & (dfLS['AUV_Northing'] != 0)]
    df_Full_Dive = df_Full_Dive[(df_Full_Dive['AUV_Easting'] != 0) & (df_Full_Dive['AUV_Northing'] != 0)]
    
    df1 = dfLS[dfLS['DepthFlag'] == 1]
    df1 = df1.copy()  
    df0 = dfLS[dfLS['DepthFlag'] == 0]

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(df_Full_Dive['Date_Time'], df_Full_Dive['AUV_Depth'], '--', label='Depth of AUV', markersize=0.5, color='gray')
    plt.plot(df_Full_Dive['Date_Time'], df_Full_Dive['AUV_Depth']-df_Full_Dive['AUV_Altitude'], 'k', label='Seafloor')
    plt.plot(df0['Date_Time'], df0['AUV_Depth'], '.', label='Bad Data Point', markersize=1.5, color='orange')
    plt.plot(df1['Date_Time'], df1['AUV_Depth'], '.g', label='Sensor Reached Seafloor', markersize=1.5)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xlabel(f'Time on {dfLS["Date_Time"].iloc[0].strftime("%m/%d/%y")}')  # Use the first timestamp for the date
    plt.ylabel('Depth (m)')
    plt.grid()
    # plt.title('AUV Depth Profile with Good and Bad Data Points')
    print(plot_dir)
    print(DIVE_NAME)
    plt.savefig(os.path.join(plot_dir, f'{DIVE_NAME}_AUV_Dive_Profile.png'))
    plt.close()
    log_message("Generated dive profile plot")

    log_message("Generating depth histogram...")
    # Histogram of depth values
    plt.figure(figsize=(8.5, 5.5))
    plt.hist(df1['Depth'], bins=150, alpha=0.7, color='blue', density=True)
    plt.xlabel('Depth (m)')
    plt.ylabel('Probability Density')
    # plt.title('Histogram of Good Btm Depth Values')
    # plt.xlim(0, -150)
    plt.grid()
    plt.savefig(os.path.join(plot_dir, f'{DIVE_NAME}_AUV_Depth_Histogram.png'))
    plt.close()
    log_message("Generated depth histogram")

    log_message("Generating intensity analysis plots...")
    # plot histogram and depth to intensity scatter plot
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 5.5))
    axs[0].hist(df1['AverageIntensityAll'], bins=150, range=(0,1500), alpha=0.7, label='Intensity All', color='blue', density=True)
    axs[0].hist(df1['AverageIntensityBtm'], bins=150, range=(0,1500), alpha=0.7, label='Intensity Btm', color='red', density=True)
    axs[0].axvline(x=MIN_INTENSITY_THRESHOLD, color='black', linestyle='--', label='Intensity Threshold')
    axs[0].set_xlabel('Intensity')
    axs[0].set_ylabel('Probability Density')
    # axs[0].set_title('Histogram of Intensity Values')
    axs[0].legend()
    axs[0].grid()
    axs[1].set_xlim(0, 1500)

    df1a = df1[df1['Depth_Approach'] == 1]
    df1b = df1[df1['Depth_Approach'] == 2]
    axs[1].scatter(df1a['AUV_Altitude'], df1a['AverageIntensityAll'], label='High Intensity Area', color='darkorange', s=2)
    axs[1].scatter(df1b['AUV_Altitude'], df1b['AverageIntensityAll'], label='Low Intensity Area', color='darkcyan', s=2)
    axs[1].set_xlabel('Depth (m)')
    axs[1].set_ylabel('Intensity')
    # axs[1].set_title('Depth vs Intensity')
    axs[1].legend()
    axs[1].grid()
    axs[1].set_xlim(0, 10)
    axs[1].set_ylim(0, 1600)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{DIVE_NAME}_AUV_Depth_vs_Intensity.png'))
    plt.close()
    log_message("Generated intensity analysis plots")

    log_message("Generating position maps...")
    # Position plot
    plt.figure(figsize=(8.5, 5.5))
    plt.scatter(df0['AUV_Easting'], df0['AUV_Northing'], alpha=0.5, label='Bad Data Point', color='orange', s=1.5)
    plt.scatter(df1['AUV_Easting'], df1['AUV_Northing'], alpha=0.5, label='AUV Position', color='green', s=1.5)
    plt.plot(df_Full_Dive['AUV_Easting'], df_Full_Dive['AUV_Northing'], 'k', linewidth=0.25, label='AUV Path')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    # plt.title('AUV Position')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, f'{DIVE_NAME}_AUV_Position.png'))
    plt.close()

    # Scatter plot of AUV position with depth color coding
    plt.figure(figsize=(8.5, 5.5))
    plt.plot(df_Full_Dive['AUV_Easting'], df_Full_Dive['AUV_Northing'], 'k', linewidth=0.25, label='AUV Path', zorder=1)
    scatter = plt.scatter(df1['AUV_Easting'], df1['AUV_Northing'], c=df1['Depth'], cmap='viridis', zorder=2)
    plt.colorbar(scatter, label='Depth (m)')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    # plt.title('AUV Position with Depth Color Coding')
    plt.grid()
    plt.savefig(os.path.join(plot_dir, f'{DIVE_NAME}_AUV_Position_Depth_Color.png'))
    plt.close()

    log_message("Generating depth deviation analysis...")
    # Position with depth deviation color coding
    dfnan = dfLS.copy()
    dfnan.loc[dfLS['DepthFlag'] == 0, 'Depth'] = np.nan
    Avg_Velocity = dfnan['AUV_Velocity'].mean()
    Smooth_Distance = 5 # meters
    dt= dfnan['Date_Time'].diff().dt.total_seconds().mean()  # Convert to microseconds
    smooth_timestep = int(round((Smooth_Distance / Avg_Velocity)/dt )) # Convert to microseconds
    dfnan['Smoothed_Depth'] = dfnan['Depth'].rolling(window=smooth_timestep, min_periods=smooth_timestep, center=True).mean()
    dfnan['Deviation_Depth'] = dfnan['Depth'] - dfnan['Smoothed_Depth']
    # sort by Deviation_Depth
    dfnan.sort_values(by='Deviation_Depth', inplace=True)
    dfnan.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(df_Full_Dive['AUV_Easting'], df_Full_Dive['AUV_Northing'], 'k', linewidth=0.25, label='AUV Path', zorder=1)
    scatter = plt.scatter(dfnan['AUV_Easting'], dfnan['AUV_Northing'], c=dfnan['Deviation_Depth'], cmap='coolwarm',
                          vmin=-3, vmax=3, zorder=2)
    plt.colorbar(scatter, label='Depth Deviation (m)')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    # plt.title('AUV Position with Depth Color Coding')
    plt.grid()
    plt.savefig(os.path.join(plot_dir, f'{DIVE_NAME}_AUV_Position_Depth_Deviation_Color.png'))
    plt.close()


    # Scatter plot of AUV position with intensity color coding
    df1.sort_values(by='AverageIntensityBtm', inplace=True)
    df1.reset_index(drop=True, inplace=True)   
    plt.figure(figsize=(8.5, 5.5))
    plt.plot(df_Full_Dive['AUV_Easting'], df_Full_Dive['AUV_Northing'], 'k', linewidth=0.25, label='AUV Path', zorder=1)
    scatter = plt.scatter(df1['AUV_Easting'], df1['AUV_Northing'], c=df1['AverageIntensityBtm'], cmap='viridis', 
                          vmin=50, vmax=1500, zorder=2)
    plt.colorbar(scatter, label='Intensity')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    # plt.title('AUV Position with Intensity Color Coding')
    plt.grid()
    plt.savefig(os.path.join(plot_dir, f'{DIVE_NAME}_AUV_Position_Intensity_Color.png'))
    plt.close()
    
    log_message("All summary plots generated successfully")

def Step01_Find_Good_Data(BaseDir, MIN_INTENSITY_THRESHOLD, BAD_POINT_THRESHOLD, RADIUS, xyz_files=None, log_callback=None):
    """
    Main function to process LLS data and generate summary plots.
    :param BaseDir: Base directory containing the data.
    :param MIN_INTENSITY_THRESHOLD: anything above this is considered a good point, used to keep points in the radius, and build percent of LLS points considered to be good (default is 100)
    :param BAD_POINT_THRESHOLD: percent of points that need to be above MIN_INTENSITY_THRESHOLD to use basic bottom find, if BAD_POINT_THRESHOLD % of the points are less then MIN_INTENSITY_THRESHOLD, use an alternate approach to look for the bottom (default is 70)
    :param RADIUS:  draws a circle around the AUV and rejects low intensity points in the radius, there can be a lot of noise and bad points near the sensor beam edges, (default is 4m) 
    :param xyz_files: dont need to include, if empty it runs all LLS*.xyz files in the LLS dir, or you can specify certain XYZ files
    :param log_callback: Optional callback function for logging messages to GUI
    """
    
    # Helper function to log to both terminal and GUI
    def log_message(message):
        print(message)  # Always print to terminal
        if log_callback:
            log_callback(message)  # Also send to GUI if callback provided
    
    DiveNumber=BaseDir.split('\\')[-1]
    LLSDir=os.path.join(BaseDir, 'LLS')
    LLSOutputDir=os.path.join(BaseDir, 'LLS_Output')
    VehicleDir=os.path.join(BaseDir, 'Vehicle_Data')
    VehicleOutputDir=os.path.join(BaseDir, 'Vehicle_Output')
    ImageDir=os.path.join(BaseDir, 'Images')
    ImageOutputDir=os.path.join(BaseDir, 'Images_Output')

    if not os.path.exists(VehicleOutputDir):
        os.makedirs(VehicleOutputDir)
    if not os.path.exists(ImageOutputDir):
        os.makedirs(ImageOutputDir)
    if not os.path.exists(LLSOutputDir):
        os.makedirs(LLSOutputDir)

    # find Phins compile data
    log_message("Checking Phins navigation data...")
    phins.phins_check(VehicleDir)

    if xyz_files is None:
        # optional to specify XYZ files, otherwise find all LLS_*.xyz files
        xyz_files = glob.glob(os.path.join(LLSDir, 'LLS_*.xyz'))
    else:
        # ensure xyz_files is a list of full paths
        xyz_files = [os.path.join(LLSDir, f) for f in xyz_files if f.endswith('.xyz')] 

    # check for dir and files
    LLS_Check(LLSDir, LLSOutputDir, xyz_files, log_callback)

    # estimate run time
    total_size, Start_Time = Estiamted_Run_Time(xyz_files, LLSDir, process_speed=4.53, log_callback=log_callback)

    dfLS = pd.DataFrame(columns=['FileNumber', 'FileName', 'Date_Time', 'TimeUnix', 'TotalPoints', 'NumLIpts', 'NumSLpts',
                                 'PercentLIPoints', 'AUV_Heading', 'AUV_Pitch', 'AUV_Roll', 'AUV_Depth', 'AUV_Altitude', 'AUV_Easting',
                                 'AUV_Northing','AUV_Velocity', 'Depth_Approach', 'Depth', 'Depth25', 'Depth75', 'DepthMax', 'DepthMin', 'DepthStd',
                                 'AverageIntensityAll', 'MaxIntensityAll', 'MinIntensityAll', 'StdIntensityAll',
                                 '25thPercentIntensityAll', '75thPercentIntensityAll', 'AverageIntensityBtm','DepthFlag',
                                 'PointDensity'])

    rows_list = []
    file_count = 0
    log_message(f"Processing {len(xyz_files)} LLS files...")
    
    for file in xyz_files:
        file_count += 1
        log_message(f"Processing file {file_count}/{len(xyz_files)}: {os.path.basename(file)}")
        file_rows = process_xyz_file(file, file_count, RADIUS, MIN_INTENSITY_THRESHOLD, 
                                BAD_POINT_THRESHOLD, VehicleDir, log_callback)
        rows_list.extend(file_rows)

    dfLS = pd.DataFrame(rows_list)
    dfLS['Date_Time'] = pd.to_datetime(dfLS['Date_Time'])
    dfLS.sort_values(by='Date_Time', inplace=True)
    dfLS.reset_index(drop=True, inplace=True)
    # Save the DataFrame to a CSV file
    output_file = os.path.join(LLSOutputDir, 'Processed_LLS.csv')
    dfLS.to_csv(output_file, index=False)
    log_message(f"Saved processed LLS data to {output_file}")

    # Process file summaries
    log_message("Creating file summaries...")
    dfLS_Files = pd.DataFrame(columns=['FileName', 'TimeStart', 'TimeEnd', 'PercentGood', 'AUV_Heading', 'AUV_Pitch', 'AUV_Roll', 'AUV_Depth', 'AUV_Altitude',
                                    'AUV_Easting', 'AUV_Northing', 'AUV_Velocity','SeafloorDepthAvg', 'SeafloorDepthMax', 'SeafloorDepthMin', 'SeafloorDepthStd',
                                    'IntensityAllAvg', 'IntensityAllMax', 'IntensityAllMin', 'IntensityAllStd',
                                    'AverageIntensityBtm'])
    
    file_ids= dfLS['FileName'].unique()
    time_start_list = []
    dfLS_Files_rows = []
    Total_Voyis_Time = 0
    for file_id in file_ids:
        file_data = dfLS[dfLS['FileName'] == file_id]
        TimeStart = datetime.datetime.fromtimestamp(file_data['TimeUnix'].min()/1000000, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')
        time_start_list.append(TimeStart)
        TimeEnd = datetime.datetime.fromtimestamp(file_data['TimeUnix'].max()/1000000, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')
        Total_Voyis_Time = Total_Voyis_Time+(file_data['TimeUnix'].max() - file_data['TimeUnix'].min()) / 1000000  # in seconds
        bounding_box = [file_data['AUV_Easting'].min()-3, file_data['AUV_Easting'].max()+3, file_data['AUV_Northing'].min()-3, file_data['AUV_Northing'].max()+3]
        PercentGood = (file_data['DepthFlag'].sum()/len(file_data)*100)
        row_dict = {'FileName': file_id,
                'TimeStart': TimeStart, 
                'TimeEnd': TimeEnd,
                'PercentGood' : PercentGood,
                'AUV_Heading': file_data['AUV_Heading'].mean(),
                'AUV_Pitch': file_data['AUV_Roll'].mean(),
                'AUV_Roll': file_data['AUV_Roll'].mean(),
                'AUV_Depth': file_data['AUV_Depth'].mean(),
                'AUV_Altitude': file_data['AUV_Altitude'].mean(),
                'AUV_Easting': file_data['AUV_Easting'].mean(),
                'AUV_Northing': file_data['AUV_Northing'].mean(),
                'AUV_Velocity': file_data['AUV_Velocity'].mean(),
                'SeafloorDepthAvg': file_data['Depth'].mean(),
                'SeafloorDepthMax': file_data['Depth'].max(),
                'SeafloorDepthMin': file_data['Depth'].min(),
                'SeafloorDepthStd': file_data['Depth'].std(),
                'IntensityAllAvg': file_data['AverageIntensityAll'].mean(),
                'IntensityAllMax': file_data['AverageIntensityAll'].max(),
                'IntensityAllMin': file_data['AverageIntensityAll'].min(),
                'IntensityAllStd': file_data['AverageIntensityAll'].std(),
                'AverageIntensityBtm': file_data['AverageIntensityBtm'].mean()}
        dfLS_Files_rows.append(row_dict)
    dfLS_Files = pd.DataFrame(dfLS_Files_rows)
    # Save the DataFrame to a CSV file
    output_file = os.path.join(LLSOutputDir, 'Processed_LLS_Files.csv')
    dfLS_Files.to_csv(output_file, index=False)
    log_message(f"Saved processed file summaries to {output_file}")
    
    Total_Run_Time = datetime.datetime.now() - Start_Time
    log_message(f"Total Run Time: {Total_Run_Time}")
    proces_rate = total_size / Total_Run_Time.total_seconds()
    log_message(f"Processing rate: {proces_rate:.2f} MB/s")

    log_message("Processing navigation surface offset...")
    StartDive, EndDive, GPS_OffsetG, GPS_OffsetP = phins.NAV_surface_offset(VehicleDir, VehicleOutputDir, pd.to_datetime(time_start_list, utc=True).mean())
       
    df_Full_Dive = pd.DataFrame()
    df_Full_Dive['Date_Time'] = pd.date_range(start=StartDive, end=EndDive, freq='1s')
    df_Full_Dive['Timestamp'] = df_Full_Dive['Date_Time'].apply(lambda x: x.timestamp() if pd.notnull(x) else None)
    df_Full_Dive = phins.add_IMU_NAV(df_Full_Dive, VehicleDir)

    log_message("Creating summary plots...")
    Summary_plots(dfLS, LLSOutputDir, DiveNumber, df_Full_Dive, MIN_INTENSITY_THRESHOLD, log_callback)
    
    # create text file with summary of run time, processing rate, file size, number of files, TotalPoints, perecent DEPTH_FLAG=1 and avg point density
    summary_file = os.path.join(LLSOutputDir, f"Voyis_QuickLook_Summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Total Run Time: {Total_Run_Time}\n")
        f.write(f"Processing rate: {proces_rate:.2f} MB/s\n")
        f.write(f"Total size of files: {total_size:.2f} MB\n")
        f.write(f"Number of files: {len(xyz_files)}\n")
        f.write(f"Total Points Processed: {dfLS['TotalPoints'].sum()}\n")
        f.write(f"Percent of Good Points (DEPTH_FLAG=1): {dfLS['DepthFlag'].sum()/len(dfLS)*100:.2f}%\n")
        f.write(f"Average Point Per File: {dfLS['TotalPoints'].mean():.2f} points per file\n")
        f.write(f"Average Point Density: {dfLS['PointDensity'].mean():.2f} points per cm^2\n")
        f.write(f"Start Dive: {StartDive}\n")
        f.write(f"End Dive: {EndDive}\n")
        f.write(f"GPS Offset Good: {GPS_OffsetG} m\n")
        f.write(f"GPS Offset Poor: {GPS_OffsetP} m\n")
        f.write(f"Total Dive Time: {(EndDive - StartDive).total_seconds() / 3600:.2f} hours\n")
        f.write(f"Total Voyis Time: {Total_Voyis_Time / 3600:.2f} hours\n")
        f.write(f"Total Voyis Time vs Total Dive Time: {Total_Voyis_Time / (EndDive - StartDive).total_seconds() * 100:.2f}%\n")
        f.write(f"Minimum Intensity Threshold: {MIN_INTENSITY_THRESHOLD}\n")
        f.write(f"Bad Point Threshold: {BAD_POINT_THRESHOLD}\n")
        f.write(f"Radius for Sidelobe Detection: {RADIUS} m\n")

    log_message(f"Summary written to {summary_file}")
    log_message("LLS processing complete!")

