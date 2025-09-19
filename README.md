# ARPOGE

## Installation

1. Add the library path to your environment (adapt if ARPOGE/ is not in your home folder):

   export LD_LIBRARY_PATH=$HOME/ARPOGE/lib/:$LD_LIBRARY_PATH

2. Add the required data files into ARPOGE/data/:
   - bias_image.fits
   - mask.fits
   - reference_image_normalized.fits
   - RM_S2KL.fits (dimensions: n_kl × n_slopes)

3. From inside ARPOGE/, build and install:

   waf configure
   waf
   waf install

---

## Run

1. Launch the simulation:

   papy sim

2. In ARPOGE/apps/, start the shared memory and GUI:

   ipython setup_shm.py
   ipython gui.py

   - Open the GUI
   - Go to the **Process** tab
   - Start both processes

3. In ARPOGE/bin/, run the real-time controllers:

   ./hrtc

   Open another terminal in the same directory and run:

   ./pix2modes_cuda

---


