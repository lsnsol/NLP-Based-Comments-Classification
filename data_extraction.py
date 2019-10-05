
'''
data extraction
'''

import re
f = open('en-development', 'r')

y=[]
XX=[]
YY=[]

for i in range(500):
    x=f.readline()
    x=list(x.split("\t"))
    z=x[1]
    y=x[2][0:-1]
    XX.append(z)
    YY.append(y)


f=open('C:\\Users\\talk2\Desktop\Industrial Training ML\dev\classifier\data_1','w')
f.write("")
f.close()
f=open('C:\\Users\\talk2\Desktop\Industrial Training ML\dev\classifier\data_1','a')
for i in XX:
    if(i.find('"')==0):
        i=i[1:-1]
        f.write(i)
    else:
        f.write(i)
    f.write("\n")
f.close()
    
f=open('C:\\Users\\talk2\Desktop\Industrial Training ML\dev\classifier\class_1','w')
f.write("")
f.close()
f=open('C:\\Users\\talk2\Desktop\Industrial Training ML\dev\classifier\class_1','a')
for i in YY:
    if(i.find(",")<>-1):
        i=re.split(',' and '"',i)
        i=i[1].split(",")
        f.write(i[0])
        f.write(" ")
        f.write(i[1])
    else: 
        f.write(i)
    f.write("\n")
f.close()
print "Data pre-processed successfully"
