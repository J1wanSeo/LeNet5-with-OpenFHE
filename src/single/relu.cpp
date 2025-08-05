// conv_bn_module.cpp
#include "conv_bn_module.h"
#include "relu.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <iostream>


Ciphertext<DCRTPoly> ApproxReLU4_square(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    // size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    // x^2

    auto x2 = cc->EvalMult(ct_x, ct_x);

    // EvalAdd 전에 level 로그 찍기
    // std::cout << "[DEBUG] sum Level: " << sum->GetLevel() << ", pt_const Level: " << pt_const->GetLevel() << std::endl;

    return x2;
}

Ciphertext<DCRTPoly> ApproxReLU4_cryptonet(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    // 0.25 + 0.5 * x + 0.125 * x^2

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

    // 0.5 * x + 0.204875 * x^2 - 0.0063896 * x^4 + 0.234606

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

Ciphertext<DCRTPoly> ApproxReLU4_Student(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    // size_t slotCount = cc->GetEncodingParams()->GetBatchSize(); 

    // Insert your own approximation below

    auto result = ct_x;

    // Insert your own approximation above
    

    return result;
}


std::vector<Ciphertext<DCRTPoly>> ApplyApproxReLU4_All(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    int mode) {

    std::vector<Ciphertext<DCRTPoly>> activated;
    for (auto& ct : ct_channels) {
        // std::cout << "[RELU] Level: " << ct->GetLevel()
        //           << ", Scale: " << ct->GetScalingFactor() << std::endl;

        Ciphertext<DCRTPoly> out;
        if (mode == 0)       out = ApproxReLU4_square(cc, ct);
        else if (mode == 1)  out = ApproxReLU4_cryptonet(cc, ct);
        else if (mode == 2)  out = ApproxReLU4_quad(cc, ct);
        else                 out = ApproxReLU4_Student(cc, ct);

        activated.push_back(out);
    }
    return activated;
}

Ciphertext<DCRTPoly> UnifiedReLU(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct, int mode) {
    // size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    if (mode == 0) {
        // Square: x^2
        return cc->EvalMult(ct, ct);
    }

    if (mode == 1) {
        // CryptoNet: 0.25 + 0.5x + 0.125x^2
        auto result = ApproxReLU4_cryptonet(cc, ct);
        return result;
    }

    if (mode == 2) {
        // Quad approximation: 0.234606 + 0.5x + 0.204875x^2 - 0.0063896x^4
        auto result = ApproxReLU4_quad(cc, ct);
        return result;
    }

    // mode 3: identity fallback (or custom user-defined function)
    return ct;
}
