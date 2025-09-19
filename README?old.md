INSTALLATION:
add $HOME/ARPOGE/lib/ to LD library path (adapt if ARPOGE/ is not in home folder) :

export LD_LIBRARY_PATH=$HOME/ARPOGE/lib/:$LD_LIBRARY_PATH

add data files to ARPOGE/data/: bias_image.fits, mask.fits, reference_image_normalized.fits and RM_S2KL.fits (n_kl * n_slopes)

then in ARPOGE/:
waf configure
waf 
waf install

RUN:
launch papy sim

then go to ARPOGE/apps/:

ipython setup_shm.py 
ipython gui.py

got to the process tab of the gui and start both processes

then go to ARPOGE/bin/:
./hrtc
open other ARPOGE/bin/ terminal
./pix2modes_cuda


