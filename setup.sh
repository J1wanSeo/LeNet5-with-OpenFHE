#!/usr/bin/env bash
# ============================
# 0) 기본 설정
# ============================
export DEBIAN_FRONTEND=noninteractive
export PATH=/opt/conda/bin:$PATH
export CC=clang
export CXX=clang++
sudo sed -i 's|kr.archive.ubuntu.com|mirror.kakao.com|g' /etc/apt/sources.list
sudo sed -i 's|security.ubuntu.com|mirror.kakao.com|g' /etc/apt/sources.list


# ============================
# 1) 필수 패키지 설치
# ============================
apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake clang ninja-build libomp-dev git wget curl bzip2 ca-certificates \
    python3 python3-pip python3-venv vim tmux htop unzip pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ============================
# 2) Miniconda 설치
# ============================
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
rm /tmp/miniconda.sh

# ============================
# 3) Conda 환경 생성
# ============================
cp environment.yml /tmp/environment.yml
# cp setup.sh /workspace/setup.sh

/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
/opt/conda/bin/conda env create -f /tmp/environment.yml
/opt/conda/bin/conda clean -afy

# ============================
# 4) Intel HEXL 설치
# ============================
# git clone https://github.com/intel/hexl.git /opt/intel-hexl
# mkdir -p /opt/intel-hexl/build
# cd /opt/intel-hexl/build
# cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
# ninja -j2
# ninja install
# ldconfig
# cd ~

# ============================
# 5) OpenFHE 빌드 및 설치
# ============================
git clone --branch v1.3.1 https://github.com/openfheorg/openfhe-development.git /opt/openfhe
mkdir -p /opt/openfhe/build
cd /opt/openfhe/build
cmake -G Ninja \
    -DWITH_OPENMP=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    ..
# -DWITH_INTEL_HEXL=ON \
# -DINTEL_HEXL_PREBUILT=OFF \
ninja -j2
ninja install
ldconfig
cd ~


# ============================
# 7) Git 레포지토리 클론
# ============================
# cd ~
# git clone https://github.com/openfheorg/openfhe-configurator.git
# cd openfhe-configurator
# (echo n; echo y) | ./scripts/configure.sh
# ./scripts/build-openfhe-development.sh

# gnome-terminal -- bash -c "echo 새 터미널에서 실행됨; exec bash"

# ============================
# 8) Swap extend
# ============================
swapoff -a
rm -f /swapfile
# 3. 새로운 swapfile 8GB 생성
sudo fallocate -l 8G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=8192

# 4. 권한 설정
sudo chmod 600 /swapfile

# 5. swap 영역으로 설정
sudo mkswap /swapfile

# 6. swap 활성화
sudo swapon /swapfile

# 7. fstab에 등록 (중복 방지 후 추가)
sudo sed -i '/\/swapfile/d' /etc/fstab
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 8. 결과 확인
echo "Swap successfully resized to:"
swapon --show
free -h

/opt/conda/bin/conda init bash

exit
