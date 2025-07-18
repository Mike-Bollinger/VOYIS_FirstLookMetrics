# MDBC MGM: Remus_Data_Extraction ----------------------------------------------

#'[Author: Noah Hunt
#'[Version: 
#'[Created: 2025-01-15
#'[Modified: 
#'[Organization:NOAA National Ocean Service (NOS) National Centers for Coastal
#'[             Ocean Science (NCCOS) Marine Spatial Ecology Division (MSE) 
#'[             Seascape Ecology and Analytics Branch
#'[Project:  Developed for the Mesophotic Deep Benthic Communities Project
#'[Description: Extracts and exports Remus data csvs. This script can be expanded
#'[to extract additional data files from the rlf exports as wanted.

# 1. Configuration -------------------------------------------------------------

# load libraries
library(tidyverse)

# set file configurations
files_config <- tibble(
  file_name = c('NAV_STATE.txt', 'CTD.txt', 'PHINS INS.txt', 'BATHY.txt'),
  output_suffix = c('Nav', 'CTD', 'Motion', 'Bathy'),
  var_names = list(
    c('secs_since_1970', 'heading_degs', 'heading_rate_degs_sec', 
      'latitude', 'longitude', 'depth_m', 'altitude_m'),
    c('latitude', 'longitude', 'conductivity', 'temperature', 
      'salinity', 'sound_speed'),
    c('latitude', 'longitude', 'time', 'depth', 'heave', 'pitch', 'roll'),
    c('latitude', 'longitude', 'mission_msecs', 'depth', 'altitude')),
  new_names = list(
    c('mission_time', 'heading', 'heading_rate', 
      'latitude', 'longitude', 'depth', 'altitude'),
    NULL,
    c('latitude', 'longitude', 'time', 'Depth', 'Heave', 'Pitch', 'Roll'),
    c('latitude', 'longitude', 'mission_msecs', 'Depth', 'Altitude')))

# set cruise and dive names
cruise <- 'Cruise'
dive <- 'Dive003'

# 2. Set directories -----------------------------------------------------------

# set base directory structure
cruise_dir <- file.path('C:', 'Users', 'Noah.Hunt', 'Documents', 'NOAA', 'MDBC', 
                      'SAS_Stuff', cruise)
data_dir <- file.path(cruise_dir, dive, 'veh_data', 'Remus_RLF_export')
output_dir <- file.path(cruise_dir, dive)

# create output directory if it doesn't exist
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 3. Data extraction function -------------------------------------------------

extract_vehicle_data <- function(data_dir, file_name, var_names, new_names = NULL) {
  file_path <- file.path(data_dir, file_name)
  
  if (!file.exists(file_path)) {
    message(glue::glue("Cannot find file: {file_name}. Check directory paths and file names."))
    return(NULL)
  }
  
  # read and process data using tidyverse
  dat <- read_csv(file_path, show_col_types = FALSE) %>%
    select(all_of(var_names))
  
  # rename columns if new names provided
  if (!is.null(new_names)) {
    dat <- dat %>%
      set_names(new_names)
  }
  
  return(dat)
}

# 4. Extract all data files ---------------------------------------------------

all_data <- files_config %>%
  mutate(
    data = pmap(
      list(file_name, var_names, new_names),
      ~ extract_vehicle_data(data_dir, ..1, ..2, ..3))) %>%
  filter(!map_lgl(data, is.null))

# 5. Export CSVs --------------------------------------------------------------

# export all processed data files
all_data %>%
  select(output_suffix, data) %>%
  pwalk(~ write_csv(
    .y, 
    file.path(output_dir, paste0(dive, .x, '.csv'))))

# print summary
cat("Successfully processed and exported the following files:\n")
all_data %>%
  pull(output_suffix) %>%
  paste0(dive, ., '.csv') %>%
  cat(sep = '\n')
