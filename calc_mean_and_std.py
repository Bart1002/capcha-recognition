import numpy as np
import glob
from PIL import Image

PATHS = glob.glob('samples/*.png')

mean,std = [],[]

for p in PATHS:
    img = Image.open(p).convert('RGB')
    img = np.asarray(img).reshape(3,-1)
    mean.append(np.mean(img,axis=1))
    std.append(np.std(img,axis=1))



mean = np.mean(mean,axis=0)/255.0
std = np.mean(std,axis=0)/255.0

print(f"Mean: {mean}\nStd: {std}")