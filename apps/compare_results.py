import numpy as np
from matplotlib import pyplot as plt
import astropy.io.fits as fits
import os
from datetime import datetime
from collections import defaultdict
import re

def find_latest_folders(parent_folder):
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\w+)")

    
    latest = defaultdict(lambda: (None, None))  # type: (datetime, folder_name)

    for name in os.listdir(parent_folder):
        full_path = os.path.join(parent_folder, name)
        if not os.path.isdir(full_path):
            continue  

        match = pattern.fullmatch(name)
        if match:
            date_str, kind = match.groups()
            dt = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")

            # Update if newer
            if latest[kind][0] is None or dt > latest[kind][0]:
                latest[kind] = (dt, name)

    return {kind: folder for kind, (_, folder) in latest.items() if folder}



parent_dir = "../results/"
latest_folders = find_latest_folders(parent_dir)
results_c_dir = latest_folders['c']
results_python_dir = latest_folders['python']


python_path = os.path.join(parent_dir, results_python_dir)
c_path = os.path.join(parent_dir, results_c_dir)

loop_time_array_python = fits.getdata(os.path.join(python_path, "loop_time_array.fits"))
wfs_timestamp_array_python = fits.getdata(os.path.join(python_path, "wfs_timestamp_array.fits"))

loop_time_array_c = fits.getdata(os.path.join(c_path, "loop_time_array.fits"))
wfs_timestamp_array_c = fits.getdata(os.path.join(c_path, "wfs_timestamp_array.fits"))

print(latest_folders)  

fig, axs = plt.subplots(2, 2, figsize=(10, 6))

axs[0, 0].plot(np.diff(loop_time_array_python))
axs[0, 0].set_title("Loop time difference (Python)")

axs[0, 1].plot(np.diff(loop_time_array_c))
axs[0, 1].set_title("Loop time difference (C)")

axs[1, 0].plot(np.diff(wfs_timestamp_array_python))
axs[1, 0].set_title("WFS timestamp difference (Python)")

axs[1, 1].plot(np.diff(wfs_timestamp_array_c))
axs[1, 1].set_title("WFS timestamp difference (C)")

plt.tight_layout()
plt.show()