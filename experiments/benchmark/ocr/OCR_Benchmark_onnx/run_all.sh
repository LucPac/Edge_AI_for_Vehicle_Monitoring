#!/bin/bash

echo "============================================="
echo "              TEST MODEL ONNX                "
echo "============================================="

echo "MobileNet ONNX"
python test_mobilenet_onnx.py
echo "---------------------------------------------"

echo "ResNet ONNX"
python test_resnet_onnx.py
echo "---------------------------------------------"

echo "PP-OCRv3 ONNX"
python test_ppocrv3_onnx.py
echo "---------------------------------------------"

echo "PP-OCRv4 ONNX"
python test_ppocrv4_onnx.py
