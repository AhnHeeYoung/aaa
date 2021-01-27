def image_folder = './data/wsi/hotspot_annotation_qpdata'      // in linux for each project define image folder
def slide_name = getProjectEntry().getImageName()
//def img_path = image_folder+ slide_name[0..3]+'/' + slide_name+'.mrxs'
def img_path = image_folder + '/' + slide_name+'.svs'
print img_path
//def img_path = getCurrentImageData().getServer().getPath()
//print img_path

//def folder = new File(image_folder)
//// If it doesn't exist
//if( !folder.exists() ) {
//  // Create all folders up-to and including B
//  folder.mkdirs()
//}
dir_path = './data/wsi/'
def folder = new File(buildFilePath(dir_path, 'hotspot_annotation'))

// If it doesn't exist
if( !folder.exists() ) {
  // Create all folders up-to and including B
  folder.mkdirs()
}

def annotation_path = buildFilePath(dir_path, 'hotspot_annotation', getProjectEntry().getImageName() + '.txt')
print annotation_path
def file = new File(annotation_path)
def write_filename_count = 1
file.text = ''

// Loop through all objects & write the points to the file
for (pathObject in  getAnnotationObjects()) { // getAnnotationObjects  getAllObjects
    // Check for interrupt (Run -> Kill running script)
    if (Thread.interrupted())
        break
    // Get the ROI
    def roi = pathObject.getROI()
    def classname = pathObject.getPathClass().toString()
    if (roi == null)
        continue
    if ( write_filename_count==1){
        file << img_path << System.lineSeparator()
        write_filename_count= write_filename_count-1
        }
    // Write the points; but beware areas, and also ellipses!
    file << classname << System.lineSeparator()
    file << roi.getPolygonPoints() << System.lineSeparator()
}
print 'Done!'