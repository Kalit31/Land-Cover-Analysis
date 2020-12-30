//Collect data from Sentinel-1 SAR GRD: C-band Synthetic Aperture Radar Ground Range Detected, log scaling
var imageCollection = ee.ImageCollection("COPERNICUS/S1_GRD");

//Select required region
var geometry = ee.Geometry.Polygon(
        [[[86.61434028285362, 23.220284514128036],
          [86.61434028285362, 21.523746077955526],
          [89.03682563441612, 21.523746077955526],
          [89.03682563441612, 23.220284514128036]]], null, false),
  
//Filter out the required properties including the appropriate time          
var collectionVV = imageCollection
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filter(ee.Filter.eq('resolution', 'H'))
    .filterDate("2020-04-01", "2020-06-30")
    .filterBounds(geometry)
    .select(['VV']);

var VV = collectionVV.median();

var trueColour = { bands: ["VV"],
min: -14,
max: -7 };

//Add the layer to map
Map.addLayer(VV, trueColour, 'VV');

//Export data to Google drive
Export.image.toDrive({
image: VV,
description: 'Apr-Jun2020',
scale: 20,
region: geometry,
maxPixels: 1e9
});
