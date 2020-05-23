import os

rootDir = './test'
for dirName, subdirList, fileList in os.walk(rootDir, topdown=False):
    print('Folders found : %s' % dirName)
    print(dirName)
    for fname in fileList:
        X = str(fname)
        print(X+ " asdasd")
