sudo apt install libopenblas-dev libatlas-base-dev liblapack-dev libeigen3-dev liblapacke-dev
source ~/.env/data/bin/activate
git clone https://github.com/opencv/opencv.git --depth=1
git clone https://github.com/opencv/opencv_contrib.git --depth=1
cd opencv
mkdir build
cd build
cmake -D MAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/local/ -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python3.5/site-packages -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
make -j5
make install