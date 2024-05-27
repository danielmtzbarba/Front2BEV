# --------------------------------------------------------------------------
# BEV COLOR MAPPING Sim2Real

bev_cls_real = {

    3: {
        113:  0, # Non-Free

        125:  1, # Free space

        201:  2, # RoadLines
    },
}


# --------------------------------------------------------------------------
# BEV COLOR MAPPING CITYSCAPES TO 6K

bev_cls = {

    2: {
        0:   0, # Non-Free
        16:  0, # Vehicle
        50:  0, # Vehicle
        84:  0, # Pedestrians

        90:  1, # Free space
        190: 1, # Road Lines
        127: 1, #idk
        119: 1, # greens
        78:  1, # Cable
        153: 1, # Post
        178: 1, # TrafficLight
    },

    3: {
        0:   0, # Non-Free
        16:  0, # Vehicle
        50:  0, # Vehicle
        84:  0, # Pedestrians

        90:  1, # Free space
        190: 1, # Road Lines
        127: 1, # idk
        119: 1, # greens
        78:  1, # Cable
        153: 1, # Post
        178: 1, # TrafficLight

        120: 2, # Sidewalk
        33:  2, # islands
    },

    4: {
        0:   0, # Non-Free
        84:  0, # Pedestrians

        90:  1, # Free space
        190: 1, # Road Lines
        127: 1, # idk
        119: 1, # greens
        78:  1, # Cable
        153: 1, # Post
        178: 1, # TrafficLight

        120: 2, # Sidewalk
        33:  2, # islands

        16:  3, # Vehicle
        50:  3, # Vehicle
    },

    5: {
        0:   0, # Non-Free

        90:  1, # Free space
        190: 1, # Road Lines
        127: 1, # idk
        119: 1, # greens
        78:  1, # Cable
        153: 1, # Post
        178: 1, # TrafficLight

        120: 2, # Sidewalk
        33:  2, # islands

        16:  3, # Vehicle
        50:  3, # Vehicle

        84: 4, # Pedestrians
    },

    6: {
        0:   0, # Non-Free

        90:  1, # Free space
        127: 1, # idk
        119: 1, # greens
        78:  1, # Cable
        153: 1, # Post
        178: 1, # TrafficLight

        120: 2, # Sidewalk
        33:  2, # islands

        16:  3, # Vehicle
        50:  3, # Vehicle
        
        84: 4, # Pedestrians

       190: 5, # Road Lines

    },
}

bev_class2color = {
    0:  (150, 150, 150), # Non-Free
    1:  (255, 255, 255), # Free space
    2:  (220, 220, 220),  # Sidewalk
    3:  (  0,   7, 165),      # Vehicles
    4:  (200,  35,   0),      # Pedestrians
    5:  (255, 209, 103),    # RoadLines
}
# --------------------------------------------------------------------------
