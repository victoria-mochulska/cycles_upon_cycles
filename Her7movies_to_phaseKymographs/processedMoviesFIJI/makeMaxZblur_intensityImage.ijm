// read raw files - multi channel, multi z, multi t
// for the intensity channel (channel 1), perform max z projection, background subtraction and gaussian blurring
// save the output for each step

  //For input files specify:

InputDir=getDirectory("Choose an input directory");
new_folders=newArray("MaxProjections",  "MaxProjections_blurred_sigma3");

//For output files specify:
OutDir= InputDir



// make new folders to save processed data
for (m = 1; m < new_folders.length+1; m++) {
	new_folder_path = InputDir+new_folders[m-1];
	File.makeDirectory(new_folder_path);	
		}
		
position_counter = 1;

// read raw files
list = getFileList(InputDir+"/RawData/");
	
//for (n = 1; n < 3; n++) {
for (n = 1; n < list.length+1; n++) {
	
	if (endsWith(list[n-1], ".tif")) {
		
		print(position_counter);
		open(InputDir+"/RawData/"+list[n-1]);
	     name = File.getName(list[n-1]);
         name_final = substring(name, 0, lengthOf(name)-4); //removing .tif

       // max projection
		run("Duplicate...", "duplicate channels=1");
		close("\\Others");
		run("Z Project...", "projection=[Max Intensity] all");
		close("\\Others");
		run("Grays");
		rename("MAX_"+name_final+".tif");
		saveAs("Tiff", InputDir+"/MaxProjections/MAX_"+name_final+".tif");
	
		
		// gaussian blurring
		selectWindow("MAX_"+name_final+".tif");
		run("Gaussian Blur...", "sigma=3 stack");
		saveAs("Tiff", InputDir+"/MaxProjections_blurred_sigma3/MAX_blur3_"+name_final+".tif");
		run("Close");
		position_counter+=1;
	
}
}  