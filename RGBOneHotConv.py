import numpy as np

# 0 -> Urban  (227,50,25)
# 1 -> Rural  (175, 40, 25)
# 2 -> Agricultural crop land  (242, 242, 100)
# 3 -> Forest, Deciduous  (100, 190, 45)
# 4 -> Forest, Swamp/Mangroves  (110, 238, 160)
# 5 -> Barren Uncultivable Land (227, 75, 237)
# 6 -> Wetlands Waterbodies, Streams, Canals  (35, 50, 227)
# 7 -> Wetlands, Ponds, Lakes  (90, 150, 245)
# 8 -> Unclassified (255,255,255)


color_dict = {
    0: (227, 50, 25),
    1: (175, 40, 25),
    2: (242, 242, 100),
    3: (100, 190, 45),
    4: (110, 238, 160),
    5: (227, 75, 237),
    6: (35, 50, 227),
    7: (90, 150, 245),
    8: (255, 255, 255)
}

delta_dict = {
    0: (28, 50, 25),
    1: (25, 40, 25),
    2: (13, 13, 100),
    3: (50, 40, 45),
    4: (50, 17, 30),
    5: (28, 75, 18),
    6: (35, 50, 28),
    7: (30, 30, 10),
    8: (0, 0, 0)
}

max_color_dict = {
    0: (255, 100, 50),
    1: (200, 80, 50),
    2: (255, 255, 200),
    3: (150, 230, 90),
    4: (160, 255, 190),
    5: (255, 150, 255),
    6: (70, 100, 235),
    7: (120, 180, 255),
    8: (255, 255, 255)
}

min_color_dict = {
    0: (200, 0, 0),
    1: (150, 0, 0),
    2: (230, 230, 0),
    3: (50, 150, 0),
    4: (60, 220, 130),
    5: (200, 0, 220),
    6: (0, 0, 200),
    7: (60, 120, 235),
    8: (255, 255, 255)
}


# Convert the RGB array of shape (128,128,3) to an array of shape (16384,3)
# Now, check each of the 16384 tuples of 3 elements(R,G,B).
# Find out in which class the tuple lies by comparing with the min and max values of each class.
# Finally, return an array of shape (128,128,c). Here no of classes are 9.
def rgb_to_onehot(rgb_arr):
    num_classes = len(color_dict)
    # shape = (h,w,c). Here, h = 128, w=128, c = 9. => (128,128,9)
    shape = rgb_arr.shape[:2]+(num_classes,)
    # intialize one hot encoded array having above shape
    arr = np.zeros(shape, dtype=np.int8)
    # Reshape the RGB array from (h,w,3) => (h*w,c). (128,128,3)=>(16384,3)
    reshaped_arr = rgb_arr.reshape((-1, 3))
    # Iterate through each of the 9 classes.
    for i, clr in enumerate(color_dict):
        arr[:, :, i] = np.all(np.logical_and((reshaped_arr >= min_color_dict[i]),
                                             (reshaped_arr <= max_color_dict[i])),
                              axis=1).reshape(shape[:2])
    return arr


def onehot_to_rgb(onehot):
    # SingleLayer is a (h,w) array consisting of the index of the 1 in the onehot encoded array of 9 classes.
    single_layer = np.argmax(onehot, axis=-1)
    # Output -> (h,w,3)
    output = np.zeros(onehot.shape[:2]+(3,))
    # Iterate through all the classes and put their respective RGB values in the output.
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return np.uint8(output)
