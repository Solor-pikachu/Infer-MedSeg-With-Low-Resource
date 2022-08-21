import os
import argparse
import glob
import time

load_time = time.time()
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
softmax_helper = lambda x: F.softmax(x, 1)

import argparse

from engine.fine_network import get_fine_model
from engine.coarse_network import get_coarse_model
from engine.preprocess import crop_from_list_of_files,resample_patient
from engine.postprocess import save_segmentation_nifti_from_softmax
from engine.utils import *
from engine.utils import _compute_steps_for_sliding_window

import cc3d 
import fastremap
import threading
import os
import sys