current_dir=$(pwd)
echo "Create Development Dir"
pushd /home/mlpi 
echo "Re-permission /home/mlpi"
chown -vR mlpi:mlpi /home/mlpi 
echo "Complete Dev Dir Setup"
pushd /home/mlpi/DevOps 

echo "Fetch Pre-Built MXnet wheel"
wget https://mxnet-public.s3.amazonaws.com/install/raspbian/mxnet-1.5.0-py2.py3-none-any.whl
pip2 install mxnet-1.5.0-py2.py3-none-any.whl

echo "Fetch Inception From MXNet Models (pre-trained)"
curl --header 'Host: data.mxnet.io' --header 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:45.0) Gecko/20100101 Firefox/45.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --header 'Referer: http://data.mxnet.io/models/imagenet/' --header 'Connection: keep-alive' 'http://data.mxnet.io/models/imagenet/inception-bn.tar.gz' -o 'inception-bn.tar.gz' -L

tar -xvzf inception-bn.tar.gz
mv Inception-BN-0126.params Inception_BN-0000.params
mv Inception-BN-symbol.json Inception_BN-symbol.json

echo "Re-permission mlpi user home directory"
chown mlpi:mlpi -R /home/mlpi 


echo "Install Additional Libraries"
pip2 install boto3 
pip2 install ffmpeg-python 
pip2 install awscli
