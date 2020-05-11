from lifq.lifq_2d import Lifq_2d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from brian2 import *

imagePath = plt.imread("pics/surprised.png")
grayImg = Image.open(imagePath).convert("L")
arr = np.asarray(grayImg)
lif = Lifq_2d()
lif.fit(grayImg)
reconstr_img = lif.getDecodedSignal()



plt.title('Image after LIFQ')
plt.plot(reconstr_img, 'cmap = gray')
plt.show()


imagePath = 'pics/surprised01.png'

#
#
#plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
#plt.show()

