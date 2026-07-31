import re, os, os.path
from ij.io import DirectoryChooser
from ij.plugin import Concatenator,HyperStackConverter
from ij import IJ
from loci.plugins import BF

from java.io import File


inputFileExtension='lsm'

namePattern = re.compile('(.*DE_3.*).'+inputFileExtension) 
print(namePattern)

concatenator=Concatenator()



def concatenateWellPosition(inputDirectory, saveDirectory):

    filenames = sorted(os.listdir(inputDirectory))

    imageList=[]
    fileCount=0
    for filename in filenames:
        match = re.search(namePattern, filename)
        if (match == None):
            continue
        #print filename
        fileCount = fileCount + 1
        rawImage=BF.openImagePlus(os.path.join(inputDirectory,filename))[0]
        imageList.append(rawImage)

    result=concatenator.concatenateHyperstacks(imageList,'Concatenated hyperstack',False)
    print result.getNSlices()
    if result.getNSlices()>1:
        result2 = HyperStackConverter.toHyperStack(result, result.getNChannels(), result.getNSlices()/fileCount, fileCount, "czt", "grayscale")
        result=result2

    inputDirectoryFile=File(inputDirectory)
    parentDirName=inputDirectoryFile.getParent()
    outFileName=inputDirectoryFile.getName()

    IJ.saveAsTiff(result,os.path.join(saveDirectory,outFileName+'.tif'))

IJ.log('Concatenate script start')
inputDirectoryChooser = DirectoryChooser("Select Input Directory for many positions")
inputDirectoryGlobal=inputDirectoryChooser.getDirectory()
if inputDirectoryGlobal is None:
    sys.exit("No folder selected!")

for root, subdirectories, files in os.walk(inputDirectoryGlobal):
    #print 'root', root
    #print 'sub ', subdirectories
    #print 'files', files
    if len(subdirectories)>0:
        continue
    concatenateWellPosition(root,inputDirectoryGlobal)
IJ.log('Concatenate script end')
