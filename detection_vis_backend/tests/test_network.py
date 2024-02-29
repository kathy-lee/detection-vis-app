import os
import pytest

from detection_vis_backend.networks.network import (
    FFTRadNet, 
    RODNet_CDC, RODNet_CDCv2, RODNet_HG, RODNet_HGwI, RODNet_v2Base, RODNet_HGv2, RODNet_HGwIv2, 
    RECORD, RECORDNoLstm, RECORDNoLstmMulti, MVRECORD, 
    RADDet, 
    DAROD, 
    RAMP_CNN, 
    RadarCrossAttention, 
)