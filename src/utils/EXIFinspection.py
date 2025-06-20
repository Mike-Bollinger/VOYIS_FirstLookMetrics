from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def extract_exif_data(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                print("No EXIF data found.")
                return
            
            print("EXIF Data:")
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                print(f"{tag_name}: {value}")
            
            if 'GPSInfo' in exif_data:
                gps_info = exif_data['GPSInfo']
                print("\nGPS Data:")
                for gps_tag, gps_value in gps_info.items():
                    gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                    print(f"{gps_tag_name}: {gps_value}")
            else:
                print("No GPS data found in EXIF.")
    except Exception as e:
        print(f"Error reading EXIF data: {e}")

# Replace 'example_image.jpg' with the path to the attached image
extract_exif_data("E:\PC2403_Voyis\DIVE003\Images\ESC_stills_processed_PPS_2024-06-27T074938.458700_2170.jpg")