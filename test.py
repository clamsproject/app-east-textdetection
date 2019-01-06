import sys
from east_td import EAST_td
from datetime import datetime

st = datetime.now()
ocr = EAST_td()
a = open(sys.argv[1])
b = a.read()
c = ocr.annotate(b)
for i in c.views:
    a = i.__dict__
    print (a)
    c = a.get("contains")
    bd = a.get("annotations")
    for d in bd:
        print (d.__dict__)
print (datetime.now()-st)