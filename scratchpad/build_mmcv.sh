set -eu pipefail

docker run --gpus all --network host -v $PWD/host -it nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04 bash -ec '

apt-get -y update --fix-missing
apt-get install -y software-properties-common curl wget git unzip
add-apt-repository -y ppa:deadsnakes/ppa

apt-get install -y python3.12 python3.12-dev
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.12 get-pip.py
python3.12 -m pip install --upgrade pip
python3.12 -m pip install setuptools packaging

echo "Creating tmp directory"
TMPDIR="/tmp"
mkdir -p ${TMPDIR}/packages
cd ${TMPDIR}/packages

wget https://download.pytorch.org/whl/cu130/torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl
ls -lh
python3.12 -m pip install torch-2.9.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl
rm torch-2.9.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl

git clone https://github.com/open-mmlab/mmcv.git
echo "Moving into mmcv directory"
cd mmcv

echo "Checking out v2.2.0 && installing requirements"
git checkout v2.2.0
python3.12 -m pip install -r requirements/runtime.txt

export PATH="$PWD:$PATH"

echo "Checking nvcc version and setting CUDA Arch List"
nvcc --version
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;11.0;12.0;12.1"
export FORCE_CUDA="1"

echo "Building mmcv"
python3.12 -m pip install -e . -v

echo "Creating wheel file"
python3.12 setup.py bdist_wheel

echo "Wheel file created and moved to"
ls
ls dist/
cp dist/mmcv-2.2.0-cp312-cp312-linux_x86_64.whl /mmcv-2.2.0-cp312-cp312-torch-2.9.1+cu130-linux_x86_64.whl

echo "Saved generated wheel file at /mmcv-2.2.0-cp312-cp312-torch-2.9.1+cu130-linux_x86_64.whl"
'
