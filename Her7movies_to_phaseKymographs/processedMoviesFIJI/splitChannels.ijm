
InputDir=getDirectory("Choose an input directory");
//new_folders=newArray("betaCateninRaw_z", "pcnaRaw_z", "tlRaw_z", "betaCateninRaw_zSum"); //ch = 1, 3, 2.
new_folders = newArray("her7Raw_z", "tlRaw_z");
file_suffix = newArray("_her7_z.tif", "_tl_z.tif");;
file_prefix = "";
// -------------------------------------------------------------------------------------------------
//For output files specify:
OutDir= InputDir



// make new folders to save processed data
for (m = 1; m < new_folders.length+1; m++) {
	new_folder_path = InputDir+new_folders[m-1];
	File.makeDirectory(new_folder_path);	
		}

// read raw files
list = getFileList(InputDir+"/RawData_rotated/");

// process data
//for (n = 1; n < 5; n++) {
for (n = 1; n < list.length+1; n++) {
	
	if (endsWith(list[n-1], ".tif")) {
		
		print(n);
		open(InputDir+"/RawData_rotated/"+list[n-1]);
	    name = File.getName(list[n-1]);
        name_input = substring(name, 0, lengthOf(name)-4); //removing .tif
	    name_output = file_prefix+name_input;
	     
		run("Split Channels");
		selectWindow("C1-"+name_input+".tif");
		saveAs("Tiff", InputDir+new_folders[0]+"/"+name_output+file_suffix[0]);
		
		
		selectWindow("C2-"+name_input+".tif");
		saveAs("Tiff", InputDir+new_folders[1]+"/"+name_output+file_suffix[1]);
		
		
		//selectWindow("C3-"+name_final+".tif");
		//saveAs("Tiff", InputDir+"/pcnaRaw_z/"+name_final+file_suffix[2]);
		
		close("*");
		
	}
}