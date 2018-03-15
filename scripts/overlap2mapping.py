import numpy as np
import sys
filename = sys.argv[1]
f = open(filename, 'r')

integers = np.vectorize(lambda x : int(x))

multiple_mothers, no_mother, no_daughters = 0, 0, 0
mapping = {}
for l in f.readlines():
    m,d = l.split(' - ')
    if d.endswith('\n'):
        d = d[:-2]
    if d.endswith(' '):
        d = d[:-1]
    mother = [i for i in m.split(' ') if i != '']
    daughters = [i for i in d.split(' ') if i != '']
    if len(mother) > 1:
        multiple_mothers += 1
    if len(mother) == 0:
        no_mother += 1
    if len(daughters) == 0:
        no_daughters += 1
    if len(mother) == 1 and len(daughters) != 0:
        mapping[int(mother[0])] = integers(daughters)

out_file = filename.replace('Matching', 'Mapping')
# from vplants.tissue_analysis.lineage import lineage_to_file
from vplants.tissue_analysis.lineage import lineage_to_file
lineage_to_file(out_file, mapping)
