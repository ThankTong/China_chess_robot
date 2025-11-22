# setup.py
# Lệnh để biên dịch:
# python setup.py build_ext --inplace
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='robot_sdk', # Đổi tên module
    ext_modules=cythonize('robot_sdk_core.py') # Trỏ đến file .py đã đổi tên
)