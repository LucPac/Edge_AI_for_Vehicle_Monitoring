#!/bin/bash

echo "============================================="
echo "           TEST MODEL PADDLE OCR             "
echo "============================================="

echo "MobileNet"
python test_mobilenet.py
echo "---------------------------------------------"

echo "ResNet"
python test_resnet.py
echo "---------------------------------------------"

echo "PP-OCRv3"
python test_ppocrv3.py
echo "---------------------------------------------"

echo "PP-OCRv4"
python test_ppocrv4.py
