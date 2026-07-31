import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import com.opencsv.CSVWriter;

import ij.IJ
import ij.ImageJ
import ij.ImagePlus
import ij.gui.Overlay
import ij.gui.PlotWindow
import ij.gui.PolygonRoi
import ij.gui.ProfilePlot
import ij.gui.Roi
import ij.plugin.filter.Binary
import sc.fiji.analyzeSkeleton.AnalyzeSkeleton_
import inra.ijpb.segment.Threshold
import org.apache.commons.io.FilenameUtils
import ij.plugin.frame.RoiManager // sapna

import groovy.io.FileType

// INPUT UI

//#@ File (label="Midline mask file") inputLabelFile
//#@ File (label="Intensity image file") inputImageFile
//#@ Double (label = "Tail width parameter", value=3) roiScalingFactor
//#@ Boolean (label="Show intermediate results", default="false") showIntermediateResults
//#@ Boolean (label="Save results", default="true") saveResults


//outputDir ="/Volumes/sapna4tb/20200304_medaka/tailImaging/Carina_data_21-27C_Her7/roiValues_bp/"

///////
masterDir = "/Volumes/sapna4tb/20200304_medaka/tailImaging/chronic_27C/pooledAnalyses/analysesWtihoutRegistration2/"   
intensityImagesDir = masterDir+"her7Raw_zMax_blurredSigma3/"
midlineImagesDir = masterDir+"midline/"
outputDir = masterDir+"roiValues/her7/"  

intensityImageSuffix = "_her7_zMax_blur3.tif";
midlineImageSuffix = "_midline.tif";
/////////
def samplePrefices = new String [] {'230704_P3', '230704_P5', '230722_P3', '230819_P4', '230820_P3'};
///////

def int roiScalingFactor = 45 //70 for 1024*1024
def showIntermediateResults = false
/////////////

def sampleName =  intensityImagesDir + samplePrefices[0]+ intensityImageSuffix;
println sampleName

def int nSamples = samplePrefices.size()
println nSamples


//for (i=3; i<4; i++){

for (i=0; i<nSamples; i++){
	intensityImagePath = intensityImagesDir + samplePrefices[i] + intensityImageSuffix; 
// open intensity image
	def intensityImageXYT = IJ.openImage(intensityImagePath.toString())
		if(showIntermediateResults){
    		intensityImageXYT.show()
			}
	
	midlineImagePath = midlineImagesDir + samplePrefices[i] + midlineImageSuffix; 
	// midline mask
	def midlineImageXYT = IJ.openImage(midlineImagePath.toString())
		if(showIntermediateResults){
    		midlineImageXYT.show()
			}

// threshold midline mask
	def midlineMaskXYT = Threshold.threshold(midlineImageXYT, 1, Math.pow(2, midlineImageXYT.bitDepth))
		if(showIntermediateResults){
    		midlineMaskXYT.show()
		}
		
		
	// IDEA: use skeletonization and analyze skeleton  to create line ROIs for each slice/frame
	def binary = new Binary()
	def analyzeSkeleton = new AnalyzeSkeleton_() // analyze skeleton to get line ROIs

	def extractedSliceMask = new ImagePlus()
	def extractedSliceImage = new ImagePlus()

	def nFrames = midlineMaskXYT.getNSlices()
	def coordinates = new double[nFrames][]
	def intensities = new double[nFrames][]
	def coordinates_roi_x = new double[nFrames][]
	def coordinates_roi_y = new double[nFrames][]

	for (int j = 0;j<nFrames;j++ ){ // doing only 1 frame right now, total length = midlineMaskXYT.getNSlices()
		
    	midlineMaskXYT.setPosition(j+1)
    	intensityImageXYT.setPosition(j+1)
    	extractedSliceMask = midlineMaskXYT.crop()
    	extractedSliceImage = intensityImageXYT.crop()
    	def skeleton = extractedSliceMask.duplicate()
    	def runInt = binary.setup("skel", skeleton)
    	binary.run(skeleton.getProcessor()) // skeletonize
    	analyzeSkeleton.setup("", skeleton)
    	analyzeSkeleton.displaySkeletons = true
    	def skeletonResult = analyzeSkeleton.run(0, false, true,
            extractedSliceImage, true, true) // graphs contain edges -> slabs -> pixel positions
    	def xPoints =new int[skeletonResult.graph[0].edges[0].slabs.size()]
   		def yPoints =new int[skeletonResult.graph[0].edges[0].slabs.size()]
    	for (int itr = 0; itr < skeletonResult.graph[0].edges[0].slabs.size(); itr++){ // get points from an edge
        	xPoints[itr] = skeletonResult.graph[0].edges[0].slabs[itr].x
        	yPoints[itr] = skeletonResult.graph[0].edges[0].slabs[itr].y
    	}
    
    	coordinates_roi_x[j] = xPoints //sapna
    	coordinates_roi_y[j] = yPoints //sapna
    	//println skeletonResult.graph[0].edges[0].slabs.size()

    // create a free line ROI from selection
    	def roi = new PolygonRoi(xPoints, yPoints, skeletonResult.graph[0].edges[0].slabs.size(), Roi.FREELINE)
    	roi.setStrokeWidth(roiScalingFactor) // set freehand line drawing stroke width
	//    roi = Selection.lineToArea(roi)

	
    	Overlay overlay = extractedSliceImage.getOverlay() // create overlay for intensity image
    	if (overlay==null)
        	overlay = new Overlay()
    	overlay.add(roi) // add roi to the intensity image
    	extractedSliceImage.setRoi(roi, true)
    
    	//roi.setPosition(i); sapna
    //rm.addRoi(roi) sapna
    
    	extractedSliceImage.setOverlay(overlay) // set overlay
       // extractedSliceImage.show() //---- add something to save this image
    	def profiler = new ProfilePlot(extractedSliceImage, true) // do intensity profile using roi
    	def profileData = profiler.getProfile() // profile data is ana array of intensity values along the line roi
                // NOTE: You can store this data for creating kymographs. The problem here is that we dont get the
                // calibrated distance units. However, one can get it using `List` option of the plot

    	def window = new PlotWindow(profiler.getPlot()) // plot the intensity profile
   		def rt = window.getResultsTable() // get results table out of it that contains calibrated X and Y values
    	if (showIntermediateResults){
        	rt.show(rt.getTitle())
    	}

    // Further steps (suggestions): 1) You can use `profileData` for intensity profile
    //                              2) You can use x column of `rt` to get calibrated coordinates
  //rt.save("/Users/chhbra/Desktop/Results.csv")
  
  		def col1 = rt.getColumn("X")
  		coordinates[j] = col1
  
  		def col2 = rt.getColumn("Y")
  		intensities[j] = col2
  
 
  
 // if (i>0){
  //println col1
  //println coordinates[i]
  
  //println col2
 // println intensities[i]
  
  //}
  
  //profiler.close()
  		window.close()
	}
	
	println coordinates[0]
	def filePathC = outputDir+samplePrefices[i]+"_coordinates.csv"
	CSVWriter writer1 = new CSVWriter(new FileWriter(filePathC));
	for (int j = 0; j<nFrames; j++){
		line1 = coordinates[j]
		line2 = line1.toString()
		println line2
		writer1.writeNext(line2)
	}
	writer1.close()
	
	println intensities[0]
	def filePathI = outputDir+samplePrefices[i]+"_intensities.csv"
	CSVWriter writer2 = new CSVWriter(new FileWriter(filePathI));
	for (int j = 0; j<nFrames; j++){
		lineNew1 = intensities[j]
		lineNew2 = lineNew1.toString()
		println lineNew2
		writer2.writeNext(lineNew2)
	}
	writer2.close()

	println coordinates_roi_x[0]
	def filePathC_x = outputDir+samplePrefices[i]+"_coordinates_skel_x.csv"
	CSVWriter writer3 = new CSVWriter(new FileWriter(filePathC_x));
	for (int j = 0; j<nFrames; j++){
		line1 = coordinates_roi_x[j]
		line2 = line1.toString()
		println line2
		writer3.writeNext(line2)
	}
	writer3.close()


	println coordinates_roi_y[0]
	def filePathC_y = outputDir+samplePrefices[i]+"_coordinates_skel_y.csv"
	CSVWriter writer4 = new CSVWriter(new FileWriter(filePathC_y));
	for (int j = 0; j<nFrames; j++){
		line1 = coordinates_roi_y[j]
		line2 = line1.toString()
		println line2
		writer4.writeNext(line2)
	}
	writer4.close()
			
}


















