current_dir=$(pwd)
pushd /home/mlpi/DevOps 

echo "Fetch & Stage OpenCV 3 Source Code"
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.3.0.zip
unzip opencv.zip
rm -f opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip
unzip opencv_contrib.zip
rm -f opencv_contrib.zip 


echo "Upgrading PIP"
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
python3 get-pip.py

echo "Installing Python Libs"
pip2.7 install nose cpplint==1.3.0 pylint==1.9.3 'numpy<=1.15.2,>=1.8.2' nose-timer 'requests<2.19.0,>=2.18.4' h5py==2.8.0rc1 scipy==1.0.1 boto3 pypandoc
pip3 install nose cpplint==1.3.0 pylint==2.1.1 'numpy<=1.15.2,>=1.8.2' nose-timer 'requests<2.19.0,>=2.18.4' h5py==2.8.0rc1 scipy==1.0.1 boto3 pypandoc

echo "TBB Parallelization Optimized Configuration for OpenCV 3"
#cmake -D CMAKE_BUILD_TYPE=RELEASE \
#    -D CMAKE_INSTALL_PREFIX=/usr/local \
#    -D INSTALL_PYTHON_EXAMPLES=ON \
#    -D OPENCV_EXTRA_MODULES_PATH=/home/mlpi/DevOps/opencv_contrib-3.3.0/modules \
#    -D BUILD_EXAMPLES=ON ..

cd opencv-3.3.0 
mkdir -v build && cd build 
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=/home/mlpi/DevOps/opencv_contrib-3.3.0/modules \
    -D CMAKE_CXX_FLAGS="-DTBB_USE_GCC_BUILTINS=1 -D__TBB_64BIT_ATOMICS=0" \
    -D ENABLE_VFPV3=ON \
    -D ENABLE_NEON=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_TBB=ON \
    -D BUILD_EXAMPLES=OFF ..

echo "Parallelized Compilation & Linking for OpenCV 3"
make -j4 

echo "Installation & Library Loading for OpenCV 3"
make install 
ldconfig 
echo "Re-permission /home/mlpi..."
chown -R mlpi:mlpi /home/mlpi 
cd $current_dir 
echo "Complete OpenCV Installation"
