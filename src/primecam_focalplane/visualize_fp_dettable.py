"""
This script is used to visualize the focalplane of the PrimeCam 280 GHz detector table
from a given detector table h5 file.
"""
from visualize_fp import plot_focalplane_primecam
from astropy.table import QTable
from toast.instrument import Focalplane
import matplotlib.pyplot as plt
import astropy.units as u

# hf_fulltable_file = "fp_f280_dettable.h5"
hf_trimtable_file = "dets_FP_PC280_500_w2.h5" 
dettable_stack = QTable.read(hf_trimtable_file, path='dettable_trim')

width= 1.2 * u.degree
fp_primecam =  Focalplane(
                detector_data=dettable_stack)

plot_focalplane_primecam(
    focalplane=fp_primecam,
    width=width,
    height=width,
    show_labels=False,
    outfile="fp_280gz_w2_trim_test.pdf"
)
plt.show()

