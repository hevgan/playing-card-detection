#UNIFORM CARD CHOOSER
from os import listdir
from os.path import isfile, join
import numpy as np
from collections import Counter

onlyfiles = [f for f in listdir('./') if isfile(join('./', f))]

onlyfiles = [onlyfiles[i]  for i in range(len(onlyfiles) ) if '.png' in onlyfiles[i] ]

onlyfiles = [onlyfiles[i].removesuffix(".png") for i in range(len(onlyfiles)) ]

runs = 1000
samples = 500
per_pic = 30

for tries in range(runs):
    print(tries, end=',')
    sample = []
    for _ in range(samples):
        for _ in range(per_pic):
            sample.append(np.random.choice(onlyfiles))
    assert len(Counter(sample)) == len(onlyfiles)


