#!/bin/bash

# 1. เข้าไปยัง Directory ของโปรแกรม (ป้องกันปัญหา Relative Path)
cd "$(dirname "$0")"


# 3. ตั้งค่า PYTHONPATH เพื่อให้หาโมดูลใน src เจอ
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 4. รันโปรแกรม
python src/main.py