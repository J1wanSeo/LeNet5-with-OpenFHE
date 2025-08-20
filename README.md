# LeNet5-with-OpenFHE
## conv_layer_full.cpp
    Input:
        - 
```
LENET5-WITH-OPENFHE
├── backup/
├── build/
├── lenet_weights_epoch(10)/
├── lenet_weights_epoch(10)_backup/
├── parameters_no_12/
├── pytorch_ground/
├── results/
├── src/
│   ├── conv_bn_module.cpp
│   ├── conv_bn_module.h
│   ├── fc_layer.cpp
│   ├── fc_layer.cpp.backup
│   ├── fc_layer.h
│   ├── main.cpp
│   ├── relu.cpp
│   ├── relu.h
│   ├── test.cpp
│   ├── autotest.sh
│   ├── avgpool.cpp
│   ├── conv_layer_full_bn.cpp
│   ├── matrix_vector_mult.cpp
│   ├── inner-product.cpp
│
├── CMakeLists.txt
├── CMakeLists.txt.1
├── input_image.txt
├── matrix_mult.cpp
├── matrix_mult.txt
├── README.md
├── test_fc.py
├── test_fhe.py
```

1. Docker 및 Visual Studio Code 설치
Docker 설치

## Windows: Docker Desktop for Windows 설치

    필수 구성: WSL2 활성화(권장) 또는 Hyper-V

    확인: PowerShell에서

    wsl --version


## macOS(Apple Silicon/Intel): Docker Desktop for Mac 설치

## Linux (Ubuntu 예시):

    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker $USER
    # 로그아웃/로그인 후 아래로 확인
    docker --version

# VS Code 설치 + 확장
## VS Code 설치
## 필수 확장:
- Docker (Microsoft)
- Dev Containers (Microsoft)

# Docker 실행
## Docker 데몬 기동
- Windows/macOS: Docker Desktop 실행 (트레이/메뉴바에서 고래 아이콘 확인)
- Linux:
```
sudo systemctl enable --now docker
systemctl status docker
```
# Docker build 수행
```
docker build -t openfhe-lenet5:latest .
```

# docker run --rm -it openfhe-lenet5
4. Visual Studio 접속
4-1. github git 폴더로 이동
> cd openfhe-configurator
4-2. Openfhe-Hexl 설치
> ./scripts/configure.sh
> n
> y
> ./scripts/build-openfhe-development.sh
5. python code 실행
> conda activate py_3_10
> cd LeNet5-with-Openfhe