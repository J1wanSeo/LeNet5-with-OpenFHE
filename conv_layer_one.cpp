// OpenFHE convolution layer (UniHENN 스타일) 구현 예시
#include "openfhe.h"
#include <vector>
#include <iostream>

using namespace lbcrypto;

// Conv2D 암호 연산 함수 (UniHENN 스타일)
Ciphertext<DCRTPoly> Conv2D_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::vector<double>& filter,
    size_t inputH,
    size_t inputW,
    size_t filterH,
    size_t filterW,
    double bias,
    PublicKey<DCRTPoly> pubkey,
    const std::vector<int>& rot_indices
) {
    std::vector<Ciphertext<DCRTPoly>> rotatedCts;

    for (size_t dy = 0; dy < filterH; dy++) {
        for (size_t dx = 0; dx < filterW; dx++) {
            size_t idx = dy * filterW + dx;
            int rotAmount = dy * inputW + dx;
            auto rotated = cc->EvalRotate(ct_input, rotAmount);

            std::vector<double> wt(inputH * inputW, 0.0);
            wt[0] = filter[idx]; // weight를 한 곳에만 위치시키는 UniHENN 방식
            auto pt_w = cc->MakeCKKSPackedPlaintext(wt);
            auto ct_mul = cc->EvalMult(rotated, pt_w);
            rotatedCts.push_back(ct_mul);
        }
    }

    auto ct_sum = cc->EvalAddMany(rotatedCts);

    std::vector<double> biasVec(inputH * inputW, 0.0);
    biasVec[0] = bias;
    auto pt_bias = cc->MakeCKKSPackedPlaintext(biasVec);
    auto ct_out = cc->EvalAdd(ct_sum, pt_bias);

    return ct_out;
}

int main() {
    // CKKS context 설정
    CCParams<CryptoContextCKKSRNS> params;
    params.SetRingDim(1 << 14);
    params.SetScalingModSize(40);
    params.SetBatchSize(4096);
    params.SetMultiplicativeDepth(3);

    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<int> rot_indices;
    for (int i = 0; i <= 12; i++) rot_indices.push_back(i);
    cc->EvalAtIndexKeyGen(keys.secretKey, rot_indices);

    // 입력 이미지 5x5를 flatten
    std::vector<double> img = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
	11, 12, 13, 14, 15,
	16, 17, 18, 19, 20,
	21, 22, 23, 24, 25
    };

    auto pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);

    std::vector<double> filter = {
        1, 0, 1,
        1, 0, 1,
        1, 0, 1
    };

    auto ct_conv = Conv2D_CKKS(cc, ct_img, filter, 5, 5, 3, 3, 1.0, keys.publicKey, rot_indices);
    Plaintext pt_result;
    cc->Decrypt(keys.secretKey, ct_conv, &pt_result);
    pt_result->SetLength(9); // 출력 3x3
    std::cout << "[Convolution 결과]" << std::endl;
    std::cout << pt_result << std::endl;

    return 0;
}

