from capcha_dataset import CapchaDataset

import glob
import numpy as numpy
import re

import numpy as np

PATHS = glob.glob('samples/*.png')
TARGETS  = [re.findall('[a-z0-9]+',i)[1] for i in PATHS]
