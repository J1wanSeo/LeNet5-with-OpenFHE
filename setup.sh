#!/usr/bin/env bash
# ============================
# 0) 기본 설정
# ============================
export DEBIAN_FRONTEND=noninteractive
export PATH=/opt/conda/bin:$PATH
export CC=clang
export CXX=clang++

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
    # -DWITH_INTEL_HEXL=ON \
    # -DINTEL_HEXL_PREBUILT=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    ..
ninja -j2
ninja install
ldconfig
cd ~

# ============================
# 6) Conda 환경 초기화
# ============================
/opt/conda/bin/conda init bash
source ~/.bashrc

# ============================
# 7) Git 레포지토리 클론
# ============================
# cd ~
# git clone https://github.com/openfheorg/openfhe-configurator.git
# cd openfhe-configurator
# (echo n; echo y) | ./scripts/configure.sh
# ./scripts/build-openfhe-development.sh

# gnome-terminal -- bash -c "echo 새 터미널에서 실행됨; exec bash"
exit
