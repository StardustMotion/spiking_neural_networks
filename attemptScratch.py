##
##import numpy as np
##import matplotlib.image as mpimg
##import matplotlib.pyplot as plt
##from PIL import Image
##
##rWeight = 0.2989;
##gWeight = 0.5870;
##bWeight = 0.1140;
##
### grayscale the input img (input is a np array representing rgb pixel colors)
##def rgb2gray(rgb):
##    return np.dot(rgb[...,:3], [rWeight, gWeight, bWeight])
##
##image = Image.open(imagePath).convert("L")
##
##img = mpimg.imread(imagePath)      
##grayImg = rgb2gray(img)
##
##
##
### grayImg.LIF'd
### grayImg.backLIF'd
##
##
##plt.imshow(grayImg, cmap=plt.get_cmap('gray'), vmin=0, vmax=255) # tweak vmin and vmax for some degradation
###plt.plot(grayImg, 'cmap = gray');
##
##plt.show()


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from brian2 import *

imagePath = 'pics/surprised01.png'

#grayImg = Image.open(imagePath).convert("L")
#arr = np.asarray(grayImg)
#plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
#plt.show()




daState = None
daSpike = None
daMatrix = None
daReconstr_array = None

# membrane_resistance : must be of type (ohm)
membrane_resistance = 550 * mohm;

# membrane_time_scale : must be of type (second)
membrane_time_scale=7 * ms

v_rest=0 * mV

lifEqs = """
        dv/dt =
        ( -(v-v_rest) + membrane_resistance * input_current(t, i) ) / membrane_time_scale : volt (unless refractory)"""

G = NeuronGroup(1, lifEqs)
run(100*ms)
