# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2018 CNRS - ENS Lyon - INRIA
#
#       File author(s): Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>
################################################################################

"""
This script create artificial segmented and intensity images to test the
'PIN_quantif.py' python script.
"""

from create_images import create_two_label_image
from create_images import create_two_sided_intensity_image
from create_images import create_left_sided_intensity_image
from create_images import create_right_sided_intensity_image

from timagetk.components import imsave

# - Write labelled test image:
test_label_fname = "test_seg.inr"
imsave(test_label_fname, create_two_label_image())

# - Write two-sided intensity image (PI):
test_PI_fname = "test_PI.inr"
imsave(test_PI_fname, create_two_sided_intensity_image())

# - Write left-sided intensity image (PIN1):
test_left_PIN1_fname = "test_left_PIN1.inr"
imsave(test_left_PIN1_fname, create_left_sided_intensity_image())

# - Write right-sided intensity image (PIN1):
test_right_PIN1_fname = "test_right_PIN1.inr"
imsave(test_right_PIN1_fname, create_right_sided_intensity_image())

%run ../PIN_quantif.py $test_PI_fname $test_left_PIN1_fname $test_label_fname --membrane_dist 6.
