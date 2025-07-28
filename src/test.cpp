#include "openfhe.h"
#include "conv_bn_module.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace lbcrypto;

// 테스트용 AvgPool 함수 실행
void TestAvgPool_5x5(CryptoContext<DCRTPoly> cc,
                     const PublicKey<DCRTPoly>& pk,
                     const PrivateKey<DCRTPoly>& sk) {
    size_t inputH = 5, inputW = 5;
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    // ==== 1. 평문 입력 데이터 생성 (1 ~ 25) ====
    std::vector<double> input(slotCount, 0.0);
    for (size_t i = 0; i < inputH; i++) {
        for (size_t j = 0; j < inputW; j++) {
            size_t idx = i * inputW + j;
            input[idx] = (double)(i * inputW + j + 1);
        }
    }

    std::cout << "[Input 5x5]\n";
    for (size_t i = 0; i < inputH; i++) {
        for (size_t j = 0; j < inputW; j++) {
            std::cout << std::setw(4) << input[i * inputW + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // ==== 2. 암호화 ====
    Plaintext pt_input = cc->MakeCKKSPackedPlaintext(input);
    Ciphertext<DCRTPoly> ct_input = cc->Encrypt(pk, pt_input);
    

    // ==== 3. 단계적 Rotation Key 생성 ====
    try {
        // 먼저 기본적인 rotation keys만 생성해보기
        std::vector<int> basicRotIndices = {1, -1, (int)inputW, -(int)inputW};
        std::cout << "[기본 Rotation Keys 생성]: ";
        for (auto idx : basicRotIndices) {
            std::cout << idx << " ";
        }
        std::cout << "\n";
        cc->EvalRotateKeyGen(sk, basicRotIndices);
        
        // 추가 rotation keys 생성
        std::vector<int> additionalRotIndices;
        for (int i = 2; i <= 3; i++) {
            additionalRotIndices.push_back(i);
            additionalRotIndices.push_back(-i);
            additionalRotIndices.push_back(i * (int)inputW);
            additionalRotIndices.push_back(-i * (int)inputW);
            additionalRotIndices.push_back(i * (int)inputW + 1);
            additionalRotIndices.push_back(i * (int)inputW - 1);
            additionalRotIndices.push_back(-i * (int)inputW + 1);
            additionalRotIndices.push_back(-i * (int)inputW - 1);
        }
        
        std::cout << "[추가 Rotation Keys 생성]: ";
        for (auto idx : additionalRotIndices) {
            std::cout << idx << " ";
        }
        std::cout << "\n\n";
        cc->EvalRotateKeyGen(sk, additionalRotIndices);
        
    } catch (const std::exception& e) {
        std::cout << "Rotation key 생성 중 오류: " << e.what() << "\n";
        
        // 오류 발생 시 모든 rotation keys 생성
        std::cout << "[Fallback: 모든 rotation keys 생성]\n";
        std::vector<int> allRotIndices;
        for (int i = -50; i <= 50; i++) {
            if (i != 0) allRotIndices.push_back(i);
        }
        cc->EvalRotateKeyGen(sk, allRotIndices);
    }

    // ==== 4. AvgPooling 실행 ====
    std::vector<Ciphertext<DCRTPoly>> input_channels = { ct_input };
    
    // AvgPool에 필요한 rotation indices (실제 사용되는 것들)
    std::vector<int> poolRotIndices = {1, -1, (int)inputW, -(int)inputW, 
                                       (int)inputW + 1, (int)inputW - 1,
                                       -(int)inputW + 1, -(int)inputW - 1};

    auto pooled = AvgPool2D_MultiChannel(cc, input_channels, inputH, inputW, pk, poolRotIndices);

    // ==== 5. 복호화 및 결과 출력 ====
    Plaintext pt_out;
    cc->Decrypt(sk, pooled[0], &pt_out);
    pt_out->SetLength(inputH * inputW);
    auto vec = pt_out->GetRealPackedValue();

    std::cout << "[AvgPool 결과: 2x2 stride=2]\n";
    for (size_t i = 0; i < inputH; i += 2) {
        for (size_t j = 0; j < inputW; j += 2) {
            size_t idx = i * inputW + j;
            std::cout << std::setw(10) << vec[idx] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // ==== 6. 평문 결과와 비교 ====
    std::cout << "[평문 계산값 (확인용)]\n";
    for (size_t i = 0; i + 1 < inputH; i += 2) {
        for (size_t j = 0; j + 1 < inputW; j += 2) {
            double sum = input[i * inputW + j]
                       + input[i * inputW + j + 1]
                       + input[(i + 1) * inputW + j]
                       + input[(i + 1) * inputW + j + 1];
            std::cout << std::setw(10) << sum / 4.0 << " ";
        }
        std::cout << "\n";
    }
}

// 더 간단한 테스트: 직접 구현한 AvgPool
void TestSimpleAvgPool(CryptoContext<DCRTPoly> cc,
                       const PublicKey<DCRTPoly>& pk,
                       const PrivateKey<DCRTPoly>& sk) {
    
    std::cout << "\n=== 직접 구현한 Simple AvgPool 테스트 ===\n";
    
    size_t inputH = 5, inputW = 5;
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    // 입력 데이터 생성
    std::vector<double> input(slotCount, 0.0);
    for (size_t i = 0; i < inputH; i++) {
        for (size_t j = 0; j < inputW; j++) {
            size_t idx = i * inputW + j;
            input[idx] = (double)(i * inputW + j + 1);
        }
    }

    // 암호화
    Plaintext pt_input = cc->MakeCKKSPackedPlaintext(input);
    Ciphertext<DCRTPoly> ct_input = cc->Encrypt(pk, pt_input);

    // 필요한 rotation keys만 생성
    std::vector<int> rotKeys = {1, (int)inputW, (int)inputW + 1};
    cc->EvalRotateKeyGen(sk, rotKeys);

    // 간단한 2x2 AvgPool 직접 구현 (0,0 위치만)
    auto ct1 = ct_input;  // (0,0)
    auto ct2 = cc->EvalRotate(ct_input, 1);  // (0,1)
    auto ct3 = cc->EvalRotate(ct_input, (int)inputW);  // (1,0)
    auto ct4 = cc->EvalRotate(ct_input, (int)inputW + 1);  // (1,1)

    // 덧셈
    auto sum = cc->EvalAdd(ct1, ct2);
    sum = cc->EvalAdd(sum, ct3);
    sum = cc->EvalAdd(sum, ct4);

    // 4로 나누기 (0.25 곱하기)
    std::vector<double> quarter(slotCount, 0.25);
    Plaintext pt_quarter = cc->MakeCKKSPackedPlaintext(quarter);
    auto result = cc->EvalMult(sum, pt_quarter);

    // 복호화 및 결과 출력
    Plaintext pt_result;
    cc->Decrypt(sk, result, &pt_result);
    pt_result->SetLength(25);
    auto vec = pt_result->GetRealPackedValue();

    std::cout << "[Simple AvgPool 결과 (0,0 위치)]: " << vec[0] << "\n";
    std::cout << "[예상값]: " << (1.0 + 2.0 + 6.0 + 7.0) / 4.0 << "\n";
}

int main() {
    // 1. CryptoContext 생성
    CCParams<CryptoContextCKKSRNS> params;
    params.SetRingDim(1 << 16);
    params.SetBatchSize(4096);
    params.SetScalingModSize(40);
    params.SetMultiplicativeDepth(5);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // 2. AvgPool 테스트 실행
    TestAvgPool_5x5(cc, keys.publicKey, keys.secretKey);
    
    // 3. 간단한 테스트도 실행
    TestSimpleAvgPool(cc, keys.publicKey, keys.secretKey);

    return 0;
}