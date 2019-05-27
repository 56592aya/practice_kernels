import pandas as pd
import numpy as np
import urllib
import chardet
import argparse
import sys
from chardet.universaldetector import  UniversalDetector



def figure_encoding(file_addr):
    
    usock = open(file_addr, mode='rb')
    detector = UniversalDetector()
    for line in usock.readlines():
        detector.feed(line)
        if detector.done:
            break;

    detector.close()
    usock.close()
    inferred_encoding = detector.result
    print(inferred_encoding)
    return (inferred_encoding)