#!/home/arashyazdiha/anaconda3/bin/python
import pandas as pd
import numpy as np
import urllib
import chardet
import argparse
import sys
from chardet.universaldetector import  UniversalDetector



def main():
    parser = argparse.ArgumentParser(description='Inferring file encoding')
    parser.add_argument('--file_addr',  help='file address to figure encoding')

    args = parser.parse_args()
    
    file_addr = args.file_addr
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

if __name__ == '__main__':
    main()