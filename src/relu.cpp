// conv_bn_module.cpp
#include "conv_bn_module.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <iostream>


Ciphertext<DCRTPoly> ApproxReLU4_square(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    // size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    auto x2 = cc->EvalMult(ct_x, ct_x);

    // EvalAdd 전에 level 로그 찍기
    // std::cout << "[DEBUG] sum Level: " << sum->GetLevel() << ", pt_const Level: " << pt_const->GetLevel() << std::endl;

    return x2;
}

Ciphertext<DCRTPoly> ApproxReLU4_cryptonet(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    auto pt_coeff0   = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.25));
    auto pt_coeff1  = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.5));
    auto pt_coeff2  = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.125));

    auto x1 = cc->EvalMult(ct_x, pt_coeff1);
    auto x2_raw = cc->EvalMult(ct_x, ct_x);
    auto x2 = cc->EvalMult(x2_raw, pt_coeff2);

    auto sum = cc->EvalAdd(x1, x2);

    // EvalAdd 전에 level 로그 찍기
    // std::cout << "[DEBUG] sum Level: " << sum->GetLevel() << ", pt_const Level: " << pt_const->GetLevel() << std::endl;

    auto result = cc->EvalAdd(sum, pt_coeff0);
    return result;
}

Ciphertext<DCRTPoly> ApproxReLU4_quad(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize(); 

    auto pt_half    = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.5));
    auto pt_coeff2  = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.204875));
    auto pt_coeff4  = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, -0.0063896));
    auto pt_const   = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.234606));

    auto x1 = cc->EvalMult(ct_x, pt_half);
    auto x2_raw = cc->EvalMult(ct_x, ct_x);
    auto x2 = cc->EvalMult(x2_raw, pt_coeff2);
    auto x4_raw = cc->EvalMult(x2_raw, x2_raw);
    auto x4 = cc->EvalMult(x4_raw, pt_coeff4);

    auto sum = cc->EvalAdd(x1, x2);
    sum = cc->EvalAdd(sum, x4);

    // EvalAdd 전에 level 로그 찍기
    // std::cout << "[DEBUG] sum Level: " << sum->GetLevel() << ", pt_const Level: " << pt_const->GetLevel() << std::endl;

    auto result = cc->EvalAdd(sum, pt_const);
    return result;
}

#include "relu.h"

std::vector<Ciphertext<DCRTPoly>> ApplyApproxReLU4_All(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    int mode) {

    std::vector<Ciphertext<DCRTPoly>> activated;
    for (auto& ct : ct_channels) {
        std::cout << "[RELU] Level: " << ct->GetLevel()
                  << ", Scale: " << ct->GetScalingFactor() << std::endl;

        Ciphertext<DCRTPoly> out;
        if (mode == 0)       out = ApproxReLU4_square(cc, ct);
        else if (mode == 1)  out = ApproxReLU4_cryptonet(cc, ct);
        else                 out = ApproxReLU4_quad(cc, ct);

        activated.push_back(out);
    }
    return activated;
}

