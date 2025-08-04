// main.cpp
#include "conv_bn_module.h"
#include "relu.h"
#include "fc_layer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

using namespace std;

inline double TimeNow() {
    return chrono::duration<double>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main() {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetRingDim(1 << 16);
    params.SetScalingModSize(40);
    params.SetBatchSize(1 << 15);
    params.SetMultiplicativeDepth(20);
    params.SetScalingTechnique(FLEXIBLEAUTO);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> input_10x10(100, 0.0); // originaly 5x5
    double val = 0.01;
    for (int r = 0; r < 5; r++) {
        for (int c = 0; c < 5; c++) {
            input_10x10[r * 10 * 2 + c * 2] = val;
            val += 0.01;
        }
    }
    auto pt_input = cc->MakeCKKSPackedPlaintext(input_10x10);
    auto ct_input = cc->Encrypt(keys.publicKey, pt_input);
    std::vector<Ciphertext<DCRTPoly>> ct_inputs = {ct_input};

    // 필터 5x5, 모두 0.1
    std::vector<double> filter_3x3(9, 0.1);

    // Rotation key 생성 (필터 회전 인덱스)
    std::set<int> rotSet;
    int inputW = 10;
    int interleave = 2;
    for (int dy = 0; dy < 3; dy++) {
        for (int dx = 0; dx < 3; dx++) {
            rotSet.insert(dy * inputW * interleave + dx * interleave);
        }
    }

    auto repack2_rot = GenerateRepackRotationKeys(10, 10, 6, 6, 2); // 10 -> 3
    rotSet.insert(repack2_rot.begin(), repack2_rot.end());
    // auto repack3_rot = GenerateRepackRotationKeys(6, 6, 3, 3);
    // rotSet.insert(repack3_rot.begin(), repack3_rot.end());
    std::vector<int> rotIndices(rotSet.begin(), rotSet.end());
    cc->EvalRotateKeyGen(keys.secretKey, rotIndices);


    // CryptoContext<DCRTPoly> cc,
    // const Ciphertext<DCRTPoly>& ct_input,
    // const std::vector<double>& filter,
    // double bias,
    // size_t inputH, size_t inputW,
    // size_t filterH, size_t filterW,
    // size_t stride,
    // size_t interleave,
    // const PublicKey<DCRTPoly>& pk)

    // Conv2D 호출 (bias=0, BN 없이)
    auto ct_conv = GeneralConv2D_CKKS(cc, ct_input, filter_3x3, 0.0,
                                      10, 10, 3, 3, 1, interleave, keys.publicKey); //input is exact size of input not original one

    // 복호화 및 출력 (14x14 출력 예상)
    SaveDecryptedConvOutput(cc, keys.secretKey, {ct_conv}, 10, 10, "test_interleave_conv");

    auto ct_repack = RepackConvolutionResult(cc, ct_conv, 10, 10, 6, 6, 2);
    // ct_repack = RepackConvolutionResult(cc, ct_repack, 12, 12, 3, 3);
    SaveDecryptedConvOutput(cc, keys.secretKey, {ct_repack}, 10, 10, "test_interleave_conv_repack");

    std::cout << "Interleave conv test done." << std::endl;
    return 0;
}
