#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

int main() {
    //////////////////////////////////////
    // 1. CKKS Context 초기화
    //////////////////////////////////////
    CCParams<CryptoContextCKKSRNS> params;
    // params.SetRingDim(1 << 15);
    // params.SetScalingModSize(40);
    // params.SetBatchSize(1 << 10);
    params.SetMultiplicativeDepth(15);
    params.SetScalingTechnique(FLEXIBLEAUTO);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

  
    //////////////////////////////////////
    // 2. 키 생성
    //////////////////////////////////////
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}); // 충분히 rotate 지원

    //////////////////////////////////////
    // 3. 마스킹된 평문 슬롯 설정
    //////////////////////////////////////
    std::vector<double> plaintext(20, 0.0);
    std::vector<size_t> validIndices = {3, 7, 10};
    std::vector<double> values = {1.0, 2.0, 3.0};

    for (size_t i = 0; i < validIndices.size(); ++i) {
        plaintext[validIndices[i]] = values[i];
    }

    Plaintext pt = cc->MakeCKKSPackedPlaintext(plaintext);
    auto ct_masked = cc->Encrypt(keys.publicKey, pt);

    //////////////////////////////////////
    // 4. 압축 (Rotate + Add)
    //////////////////////////////////////
    Ciphertext<DCRTPoly> compressed = ct_masked;
    for (size_t i = 0; i < validIndices.size(); ++i) {
        int rotAmount = validIndices[i] - static_cast<int>(i); // left rotate
        auto ct_rot = cc->EvalRotate(ct_masked, rotAmount);
        compressed = cc->EvalAdd(compressed, ct_rot);
    }

    //////////////////////////////////////
    // 5. 복호화 및 확인
    //////////////////////////////////////
    Plaintext result;
    cc->Decrypt(keys.secretKey, compressed, &result);
    // result->SetLength(batchSize);

    std::cout << "=== 압축 결과 (슬롯별 출력) ===\n";
    for (size_t i = 0; i < 20; ++i) {
        std::cout << "Slot " << i << ": " << result->GetRealPackedValue()[i] << std::endl;
    }

    //////////////////////////////////////
    // 6. 기대 결과
    //////////////////////////////////////
    std::cout << "\n=== 기대값 (i번째 슬롯에 validIndices[i]의 값이 위치해야 함) ===\n";
    for (size_t i = 0; i < validIndices.size(); ++i) {
        std::cout << "Slot " << i << ": " << values[i] << std::endl;
    }

    return 0;
}
