current_dir=$(pwd)
pushd /home/mlpi/DevOps 

echo "Fetch & Stage Apache Incubator MXnet Source"
git clone https://github.com/apache/incubator-mxnet.git --recursive 
pushd incubator-mxnet 
echo "Init & Update Repository Submodules"
git submodule init 
git submodule update
mkdir -p build && pushd build
echo "Configure MXnet"
cmake \
        -DUSE_SSE=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=ON \
        -DUSE_OPENMP=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_SIGNAL_HANDLER=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -GNinja ..

echo "Build MXnet Shared Libs (Ninja)"
ninja -j$(nproc)
cd ..
pushd python 
echo "Install MXnet Python (2.7) Bindings"
pip2.7 install -e . 

echo "Installing High Level Libraries & Components"
pip2.7 install ffmpeg-python 
pip2.7 install boto3 

echo "Re-permission /home/mlpi"
chown -R mlpi:mlpi /home/mlpi 

cd $current_dir


echo "Complete MXnet Installation"
