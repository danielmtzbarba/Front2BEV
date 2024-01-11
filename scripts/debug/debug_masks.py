import matplotlib.pyplot as plt
import os
import numpy as np

from PIL import Image, ImageFile
label_path = os.path.join('/media/danielmtz/data/datasets/NuScenes/v1.0-mini/map-labels-4/0a7aef80edbc4854bcd781a3becb943c.png')
encoded_labels = Image.open(label_path)

plt.imshow(np.array(encoded_labels))
plt.show()