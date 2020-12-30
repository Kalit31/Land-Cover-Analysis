//Collect data from Sentinel-2 MSI: MultiSpectral Instrument, Level-1C
var sent2 = ee.ImageCollection("COPERNICUS/S2");

//Select required region
var geometry = ee.Geometry.Polygon(
            [[[86.71803562935315, 23.240914593327915],
            [86.71803562935315, 21.600821400860198],
            [89.14052098091565, 21.600821400860198],
            [89.14052098091565, 23.240914593327915]]], null,false);

//Filter out the required properties including the appropriate time          
var filtered_collection = sent2.filterDate("2016-12-01", "2016-12-30")
                            .filterBounds(geometry);

var image = filtered_collection.median()
var trueColour = { bands: ["B4", "B3","B2"],
                min: 0,
                max: 3000 };

// image to the map, using the visualization parameters.
Map.addLayer(image, trueColour, "true-colour image");

//Export data to Google drive
Export.image.toDrive({
image: image,
description: 'Image',
scale: 20,
region: geometry,
maxPixels: 1e9
});