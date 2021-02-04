# UAV-Image-Stitcher
Image stitcher for geotagged UAV images.

Images are pre-organised based on coordinates and capture time. The algorithm can handle large datasets without causing stack overflow by encoding information in the filenames, and only loading the bare minimum fraction of the image into the feature matching section of code (which uses crazy ram to calculate homography etc)
