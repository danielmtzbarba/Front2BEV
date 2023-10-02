# --------------------------------------------------------------------------
bev_cls = {

    2: {
        0:   0, # Non-Free
        50:  0, # Vehicle
        90:  1, # Free space
        190: 1, # Road Lines
        127: 1, #idk
        119: 1, # greens
        78:  1, # Cable
        153: 1, # Post
        178: 1, #TrafficLight
    },

    3: {
        0:   0, # Non-Free
        50:  0, # Vehicle
        90:  1, # Free space
        190: 1, # Road Lines
        127: 1, #idk
        119: 1, # greens
        78:  1, # Cable
        153: 1, # Post
        178: 1, #TrafficLight
        120: 2, # Banqueta prro
        33:  2, # islands
    },

    4: {
        0:   0, # Non-Free
        90:  1, # Free space
        190: 1, # Road Lines
        127: 1, #idk
        119: 1, # greens
        78:  1, # Cable
        153: 1, # Post
        178: 1, #TrafficLight
        120: 2, # Banqueta prro
        33: 2, # islands
        50: 3, # Vehicle
    },

    5: {
        0:   0, # Non-Free
        90:  1, # Free space
        190: 1, # Road Lines
        127: 1, #idk
        119: 1, # greens
        78:  1, # Cable
        153: 1, # Post
        178: 1, #TrafficLight
        120: 2, # Banqueta prro
        33: 2, # islands
        50: 3, # Vehicle
    },



}
bev_color2class = {
    0:   0, # Non-Free
    50: 0, # Vehicle
    90:  1, # Free space
    190: 1, # Road Lines
    127: 1, #idk
    119: 1, # greens
    78: 1, # Cable
    153: 1, # Post
    178: 1, #TrafficLight
    120: 2, # Banqueta prro
    33: 2, # islands

}

bev_class2color = {
    0:  50, # Non-Free
    1:  255, # Free space
    2: 128, # Road Lines
}
# --------------------------------------------------------------------------