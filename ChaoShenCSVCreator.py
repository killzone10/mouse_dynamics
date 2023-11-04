import struct
import pandas as pd
import os

# Define the DATA_MOUSE structure format
data_mouse_format = "iiIII"

# Event code to event name mapping
event_mapping = {
    512: "MouseEvent(WM_MOUSEMOVE)",
    513: "Click()",
    514: "MouseEvent(WM_LBUTTONUP)",
    515: "DblClick()",
    516: "RightClick()",
    517: "MouseEvent(WM_RBUTTONUP)",
    518: "RightDblClick()",
    519: "MiddleClick()",
    520: "MouseEvent(WM_MBUTTONUP)",
    521: "MiddleDlbClick()",
}

# Folder containing .ef files
base_path = 'shen-continuous\\data'
data_folder = os.listdir(base_path)   # Replace with the correct path to your folders
basic_save_path = "ChaoShenCSV\\"
if not os.path.exists(basic_save_path):
    os.makedirs(basic_save_path)
# Initialize a dictionary to store data for each prefix
data_dict = {}

# Iterate through the .ef files in the folder
for dir in data_folder:
    combined_path = os.path.join(base_path, dir)
    save_path = os.path.join(basic_save_path, dir)
    # Check if the directory already exists; if not, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for ef_file in os.listdir(combined_path):
        if ef_file.endswith('.ef'):
            ef_file_path = os.path.join(combined_path, ef_file)
            with open(ef_file_path, "rb") as file:
                data_list = []
                while True:
                    data = file.read(struct.calcsize(data_mouse_format))
                    if not data:
                        break
                    x, y, ncode, TimeStamp, ProcessNo = struct.unpack(data_mouse_format, data)

                    # Translate the event code to event name
                    event_name = event_mapping.get(ncode, "Unknown")

                    data_list.append({
                        "x": x,
                        "y": y,
                        "ncode": ncode,
                        "event": event_name,
                        "TimeStamp": TimeStamp,
                        "ProcessNo": ProcessNo
                    })

            # Determine the prefix (e.g., '1_', '2_') from the file name
            prefix = ef_file.split('_')[0]
            if prefix == ef_file:
                prefix = ef_file.split('.')[0]

            if prefix not in data_dict:
                data_dict[prefix] = []

            data_dict[prefix].extend(data_list)

# Create Pandas DataFrames and save to separate CSV files
    for prefix, data_list in data_dict.items():
        df = pd.DataFrame(data_list)
        csv_filename = os.path.join(save_path,f'combined_data_{prefix}.csv')
        df.to_csv(csv_filename, index=False)
        print(f"Combined data for {prefix} files saved to '{csv_filename}'")
    data_dict = {}