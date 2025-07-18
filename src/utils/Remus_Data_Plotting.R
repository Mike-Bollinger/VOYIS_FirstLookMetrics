# MDBC MGM: Remus_Data_Plotting ------------------------------------------------

#'[Author: Noah Hunt
#'[Version: 
#'[Created: 2025-02-03
#'[Modified: 
#'[Organization:NOAA National Ocean Service (NOS) National Centers for Coastal
#'[             Ocean Science (NCCOS) Marine Spatial Ecology Division (MSE) 
#'[             Seascape Ecology and Analytics Branch
#'[Project:  Developed for the Mesophotic Deep Benthic Communities Project
#'[Description: Generates plots of Remus vehicle data.

# 1. Configuration -------------------------------------------------------------
# load libraries
library(tidyverse)
library(sf)
library(viridis)
library(ggpubr)
library(scales)
library(lubridate)

# set cruise and dive names
cruise <- 'Cruise'
dive <- 'Dive003'

# directory structure
cruise_dir <- file.path('C:', 'Users', 'Noah.Hunt', 'Documents', 'NOAA', 
                      'MDBC', 'SAS_Stuff', cruise)
dive_dir <- file.path(cruise_dir, dive)
dive_output_dir <- file.path(dive_dir, 'Report_Plots')
cruise_output_dir <- file.path(cruise_dir, 'cruise_outputs')

# create directories
walk(c(dive_output_dir, cruise_output_dir), ~dir.create(.x, 
                                                        recursive = TRUE, 
                                                        showWarnings = FALSE))

# 2. Utility functions ---------------------------------------------------------

deg2rad <- function(degrees) degrees * pi / 180
rad2deg <- function(radians) radians * 180 / pi
convert_heading <- function(heading) ifelse(heading > 180, heading - 360, heading)

calculate_cog <- function(nav_dat) {
  if (!all(c('latitude', 'longitude') %in% names(nav_dat))) {
    stop('The nav_dat frame must contain "latitude" and "longitude" columns.')
  }
  
  nav_dat %>%
    mutate(
      lat1 = deg2rad(latitude),
      lon1 = deg2rad(longitude),
      lat2 = deg2rad(lead(latitude)),
      lon2 = deg2rad(lead(longitude)),
      delta_lon = lon2 - lon1,
      x = sin(delta_lon) * cos(lat2),
      y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon),
      initial_cog = atan2(x, y),
      cog = ((rad2deg(initial_cog) + 180) %% 360) - 180) %>%
    select(-lat1, -lon1, -lat2, -lon2, -delta_lon, -x, -y, -initial_cog)
}

calculate_crabbing <- function(cog, heading) {
  heading <- ((heading + 180) %% 360) - 180
  crab <- cog - heading
  ((crab + 180) %% 360) - 180
}

# 3. Plotting functions -------------------------------------------------------

create_base_theme <- function() {
  theme_dark() +
    theme(
      plot.margin = margin(0.1, 0.1, 0.1, 0.1, 'cm'),
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12),
      axis.text = element_text(size = 10),
      plot.title = element_text(size = 10),
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 10),
      legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 0.5))
}

plot_motion_map <- function(data, plot_var, plot_color, bbox, 
                            remove_x_axis_title = FALSE, 
                            remove_y_axis_title = FALSE) {
  p <- ggplot(data) +
    geom_sf(size = 0.15, aes(color = .data[[plot_var]])) +
    create_base_theme() +
    scale_colour_viridis(
      option = plot_color,
      breaks = seq(min(data[[plot_var]], na.rm = TRUE), 
                   max(data[[plot_var]], na.rm = TRUE), length.out = 5),
      labels = round(seq(min(data[[plot_var]], na.rm = TRUE), 
                         max(data[[plot_var]], na.rm = TRUE), length.out = 5))) +
    scale_x_continuous(labels = label_number(accuracy = 0.01)) + 
    scale_y_continuous(labels = label_number(accuracy = 0.01)) +
    coord_sf(xlim = c(bbox['xmin'], bbox['xmax']), ylim = c(bbox['ymin'], bbox['ymax'])) +
    labs(x = 'Longitude', y = 'Latitude', color = paste0(plot_var, ' (°)')) +
    theme(legend.position = 'bottom')
  
  if (remove_x_axis_title) p <- p + theme(axis.title.x = element_blank())
  if (remove_y_axis_title) p <- p + theme(axis.title.y = element_blank())
  
  return(p)
}

plot_motion_timeseries <- function(data, plot_var, time_var, depth_var, 
                                   remove_x_axis_title = FALSE, 
                                   remove_y_axis_title = FALSE) {
  p <- ggplot(data, aes(x = .data[[time_var]], y = .data[[plot_var]], 
                        color = .data[[depth_var]])) +
    geom_line() +
    scale_color_viridis_c(
      option = "mako",
      direction = -1,
      breaks = seq(0, max(data[[depth_var]], na.rm = TRUE), length.out = 5),
      labels = round(seq(0, max(data[[depth_var]], na.rm = TRUE), length.out = 5))) +
    guides(color = guide_colorbar(reverse = TRUE)) +
    labs(x = 'Time (s)', y = paste0(plot_var, ' (°)'), color = 'Depth (m)') +
    create_base_theme() +
    theme(legend.position = 'right')
  
  if (remove_x_axis_title) p <- p + theme(axis.title.x = element_blank())
  if (remove_y_axis_title) p <- p + theme(axis.title.y = element_blank())
  
  return(p)
}

plot_motion_histogram <- function(data, plot_var, plot_color, 
                                  remove_x_axis_title = FALSE, 
                                  remove_y_axis_title = FALSE) {
  stats_summary <- data %>%
    summarise(
      min_val = min(.data[[plot_var]], na.rm = TRUE),
      max_val = max(.data[[plot_var]], na.rm = TRUE),
      mean_val = mean(.data[[plot_var]], na.rm = TRUE),
      sd_val = sd(.data[[plot_var]], na.rm = TRUE))
  
  stats_text <- with(stats_summary, paste0(
    'Min: ', round(min_val, 2), '\n',
    'Max: ', round(max_val, 2), '\n',
    'Mean: ', round(mean_val, 2), '\n',
    'SD: ', round(sd_val, 2)))
  
  p <- ggplot(data, aes(x = .data[[plot_var]])) +
    geom_histogram(fill = plot_color, bins = 100) +
    labs(x = paste0(plot_var, ' (°)'), y = 'Count') +
    create_base_theme() +
    theme(legend.position = 'bottom') +
    annotate('text', x = Inf, y = Inf, label = stats_text, hjust = 1.1, vjust = 1.1, 
             size = 3.75, color = 'white')
  
  if (remove_x_axis_title) p <- p + theme(axis.title.x = element_blank())
  if (remove_y_axis_title) p <- p + theme(axis.title.y = element_blank())
  
  return(p)
}

plot_bathy_map <- function(data, plot_var, bbox, remove_x_axis_title = FALSE, 
                           remove_y_axis_title = FALSE) {
  p <- ggplot(data) +
    geom_sf(size = 0.15, aes(color = .data[[plot_var]])) +
    create_base_theme() +
    scale_x_continuous(labels = label_number(accuracy = 0.01)) + 
    scale_y_continuous(labels = label_number(accuracy = 0.01)) +
    coord_sf(xlim = c(bbox['xmin'], bbox['xmax']), ylim = c(bbox['ymin'], bbox['ymax'])) +
    scale_colour_viridis(
      option = 'mako',
      direction = -1,
      breaks = seq(min(data[[plot_var]], na.rm = TRUE), 
                   max(data[[plot_var]], na.rm = TRUE), length.out = 5),
      labels = round(seq(min(data[[plot_var]], na.rm = TRUE), 
                         max(data[[plot_var]], na.rm = TRUE), length.out = 5))) +
    labs(x = 'Longitude', y = 'Latitude', color = paste0(plot_var, ' (m)')) +
    theme(legend.position = 'bottom', aspect.ratio = 0.5)
  
  if (remove_x_axis_title) p <- p + theme(axis.title.x = element_blank())
  if (remove_y_axis_title) p <- p + theme(axis.title.y = element_blank())
  
  return(p)
}

plot_crabbing_map <- function(data, bbox, remove_x_axis_title = FALSE, 
                              remove_y_axis_title = FALSE) {
  crab_cat_colors <- c(
    'Low Port' = '#bdd7e7', 'Moderate Port' = '#6baed6', 'Extreme Port' = '#08306b',
    'Low Starboard' = '#fee6ce', 'Moderate Starboard' = '#fd8d3c', 'Extreme Starboard' = '#7f2704')
  
  data <- data %>%
    mutate(crab_index = factor(crab_index, levels = names(crab_cat_colors)))
  
  p <- ggplot(data) +
    geom_sf(size = 0.15, aes(color = crab_index)) +
    create_base_theme() +
    scale_x_continuous(labels = label_number(accuracy = 0.01)) + 
    scale_y_continuous(labels = label_number(accuracy = 0.01)) +
    coord_sf(xlim = c(bbox['xmin'], bbox['xmax']), ylim = c(bbox['ymin'], bbox['ymax'])) +
    scale_color_manual(values = crab_cat_colors, labels = names(crab_cat_colors)) +
    guides(color = guide_legend(override.aes = list(size = 4))) +
    labs(x = 'Longitude', y = 'Latitude', color = 'Crabbing Index') +
    theme(legend.position = 'bottom', aspect.ratio = 0.5)
  
  if (remove_x_axis_title) p <- p + theme(axis.title.x = element_blank())
  if (remove_y_axis_title) p <- p + theme(axis.title.y = element_blank())
  
  return(p)
}

# 4. Data import and processing ------------------------------------------------

# file paths
file_types <- c('Motion', 'Bathy', 'Nav')
file_paths <- file.path(dive_dir, paste0(dive, file_types, '.csv'))
names(file_paths) <- file_types

# import data
data_list <- map(file_paths, ~read_csv(.x, show_col_types = FALSE))

# process motion data
motion_dat <- data_list$Motion %>%
  mutate(time = hms(time)) %>%
  st_as_sf(coords = c('longitude', 'latitude'), crs = '4326')

# process nav data
nav_dat <- data_list$Nav %>%
  calculate_cog() %>%
  mutate(
    crabbing = calculate_crabbing(cog, heading),
    crab_index = cut(crabbing,
                     breaks = c(-Inf, -10, -5, 0, 5, 10, Inf), 
                     labels = c("Extreme Port", "Moderate Port", "Low Port", 
                                "Low Starboard", "Moderate Starboard", 
                                "Extreme Starboard"), 
                     right = FALSE)) %>%
  st_as_sf(coords = c('longitude', 'latitude'), crs = '4326')

# process bathy data
bathy_dat <- data_list$Bathy %>%
  st_as_sf(coords = c('longitude', 'latitude'), crs = '4326')

# 5. Generate plots ------------------------------------------------------------

# plot configuration
plot_config <- tibble(
  var = c('Heave', 'Pitch', 'Roll'),
  color = c('mako', 'magma', 'viridis'),
  hist_color = c('#a9e1fb', '#f7705c', '#44bf70'))

# get bounding boxes
motion_bbox <- st_bbox(motion_dat)
bathy_bbox <- st_bbox(bathy_dat)
nav_bbox <- st_bbox(nav_dat)

# create plots
motion_plots <- plot_config %>%
  mutate(
    map_plot = pmap(list(var, color), 
                    ~plot_motion_map(motion_dat, ..1, ..2, motion_bbox, 
                                     remove_x_axis_title = ..1 != 'Pitch',
                                     remove_y_axis_title = ..1 != 'Heave')),
    ts_plot = pmap(list(var), 
                   ~plot_motion_timeseries(motion_dat, ..1, 'time', 'Depth',
                                           remove_x_axis_title = ..1 != 'Pitch')),
    hist_plot = pmap(list(var, hist_color), 
                     ~plot_motion_histogram(motion_dat, ..1, ..2,
                                            remove_y_axis_title = ..1 != 'Heave')))
crabbing_map <- plot_crabbing_map(nav_dat %>% 
                                    filter(!is.na(crab_index)), nav_bbox)
bathy_map <- plot_bathy_map(bathy_dat, 'Depth', bathy_bbox)

# combine plots
map_plots <- ggarrange(plotlist = motion_plots$map_plot, ncol = 3, nrow = 1, 
                       align = 'hv')
hist_plots <- ggarrange(plotlist = motion_plots$hist_plot, ncol = 3, nrow = 1, 
                        align = 'hv')
ts_plots <- ggarrange(plotlist = motion_plots$ts_plot, ncol = 3, nrow = 1, 
                      align = 'hv', common.legend = TRUE, legend = 'right')

# 6. Save plots ----------------------------------------------------------------

plot_save_config <- tibble(
  plot_name = c('motion_maps', 'motion_histograms', 'motion_timeseries', 
                'bathy_map', 'crabbing_map'),
  plot_object = list(map_plots, hist_plots, ts_plots, bathy_map, crabbing_map),
  height = c(2.8, 2.5, 3, 6, 6),
  width = c(10, 10, 8, 10, 10))

plot_save_config %>%
  pwalk(~ggsave(
    filename = paste0(dive, '_', ..1, '.png'),
    plot = ..2,
    path = dive_output_dir,
    dpi = 300,
    height = ..3,
    width = ..4,
    bg = 'white'))
