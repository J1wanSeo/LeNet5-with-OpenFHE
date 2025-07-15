#include "openfhe.h"
#include <vector>
#include <iostream>
#include <iomanip>

using namespace lbcrypto;

// 2x2 Average Pooling (stride=2)
Ciphertext<DCRTPoly> AvgPool2x2_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    size_t inputH, size_t inputW,
    const PublicKey<DCRTPoly>& pubkey,
    const std::vector<int>& rot_indices,
    size_t row, size_t col
) {
    std::vector<Ciphertext<DCRTPoly>> rotatedCts;

    for (size_t dy = 0; dy < 2; dy++) {
        for (size_t dx = 0; dx < 2; dx++) {
            int rotAmount = (row + dy) * inputW + (col + dx);
            auto rotated = cc->EvalRotate(ct_input, rotAmount);

            std::vector<double> mask(inputH * inputW, 0.0);
            mask[0] = 0.25; // average
            auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);

            auto ct_mul = cc->EvalMult(rotated, pt_mask);
            rotatedCts.push_back(ct_mul);
        }
    }

    return cc->EvalAddMany(rotatedCts);
}

std::vector<double> AvgPool_FullStride2(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    size_t inputH, size_t inputW,
    const PublicKey<DCRTPoly>& pubkey,
    const std::vector<int>& rot_indices,
    const PrivateKey<DCRTPoly>& secretKey
) {
//    size_t outH = inputH / 2;
//    size_t outW = inputW / 2;
    std::vector<double> outputs;

    for (size_t i = 0; i < inputH; i += 2) {
        for (size_t j = 0; j < inputW; j += 2) {
            auto ct_out = AvgPool2x2_CKKS(cc, ct_input, inputH, inputW, pubkey, rot_indices, i, j);
            Plaintext pt;
            cc->Decrypt(secretKey, ct_out, &pt);
            pt->SetLength(1);
            outputs.push_back(pt->GetRealPackedValue()[0]);
        }
    }

    return outputs;
}

int main() {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetRingDim(1 << 14);
    params.SetScalingModSize(40);
    params.SetBatchSize(4096);
    params.SetMultiplicativeDepth(3);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<int> rot_indices;
    for (int i = 0; i <= 20; i++) rot_indices.push_back(i);
    cc->EvalAtIndexKeyGen(keys.secretKey, rot_indices);

    // 4x4 Input Image
    std::vector<double> img = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    Plaintext pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);

    auto pooled = AvgPool_FullStride2(
        cc, ct_img, 4, 4,
        keys.publicKey,
        rot_indices,
        keys.secretKey
    );

    std::cout << "[AvgPooling 결과 (2x2, stride=2)]\n";
    for (size_t i = 0; i < pooled.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << pooled[i] << " ";
        if ((i+1) % 2 == 0) std::cout << "\n";
    }

    return 0;
}
