// conv_bn_module.cpp
#include "conv_bn_module.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <iostream>

std::vector<double> LoadFromTxt(const std::string& filename) {
    std::ifstream infile(filename);
    std::vector<double> data;
    std::string content;
    std::getline(infile, content);
    std::stringstream ss(content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) data.push_back(std::stod(token));
    }
    return data;
}

std::vector<int> GenerateRotationIndices(size_t filterH, size_t filterW, size_t inputW) {
    std::set<int> rotSet;
    for (size_t dy = 0; dy < filterH; dy++)
        for (size_t dx = 0; dx < filterW; dx++)
            rotSet.insert(dy * inputW + dx);
    return std::vector<int>(rotSet.begin(), rotSet.end());
}

Ciphertext<DCRTPoly> GeneralConv2D_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::vector<double>& filter,
    double bias,
    size_t inputH, size_t inputW,
    size_t filterH, size_t filterW,
    size_t stride,
    const std::vector<int>& rotIndices,
    const PublicKey<DCRTPoly>& pk) {

    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    size_t outH = (inputH - filterH) / stride + 1;
    size_t outW = (inputW - filterW) / stride + 1;
    std::vector<Ciphertext<DCRTPoly>> partials;

    for (size_t dy = 0; dy < filterH; dy++) {
        for (size_t dx = 0; dx < filterW; dx++) {
            size_t idx = dy * filterW + dx;
            int rotAmount = dy * inputW + dx;
            auto rotated = cc->EvalRotate(ct_input, rotAmount);

            std::vector<double> combinedMask(slotCount, 0.0);
            for (size_t i = 0; i < outH; i++) {
                for (size_t j = 0; j < outW; j++) {
                    combinedMask[(i * stride) * inputW + (j * stride)] = filter[idx];
                }
            }
            auto pt_combined = cc->MakeCKKSPackedPlaintext(combinedMask);
            auto weighted = cc->EvalMult(rotated, pt_combined);
            weighted = cc->Rescale(weighted);

            partials.push_back(weighted);
        }
    }

    auto result = cc->EvalAddMany(partials);
    std::cout << "[ConvSum] Level: " << result->GetLevel()
        //   << ", Noise: " << cc->GetDepth(result)
          << ", Scale: " << result->GetScalingFactor() << std::endl;
    result = cc->Rescale(result);
    std::vector<double> biasVec(slotCount, 0.0);
    for (size_t i = 0; i < outH; i++)
        for (size_t j = 0; j < outW; j++)
            biasVec[i * inputW + j] = bias;

    auto pt_bias = cc->MakeCKKSPackedPlaintext(biasVec);
    pt_bias->SetScalingFactor(result->GetScalingFactor());
    auto ct_bias = cc->Encrypt(pk, pt_bias);
    return cc->EvalAdd(result, ct_bias);
}

Ciphertext<DCRTPoly> GeneralBatchNorm_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    double gamma, double beta,
    double mean, double var,
    size_t slotCount) {

    double eps = 1e-5;
    double a = gamma / std::sqrt(var + eps);
    double b = beta - a * mean;

    auto pt_a = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, a));
    auto pt_b = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, b));

    pt_a->SetScalingFactor(ct_input->GetScalingFactor());
    pt_b->SetScalingFactor(ct_input->GetScalingFactor());
    
    auto scaled_mul = cc->EvalMult(ct_input, pt_a);
    scaled_mul = cc->Rescale(scaled_mul);
    return cc->EvalAdd(scaled_mul, pt_b);

}

std::vector<Ciphertext<DCRTPoly>> ConvBnLayer(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_input_channels,
    const std::string& pathPrefix,
    size_t inputH, size_t inputW,
    size_t filterH, size_t filterW,
    size_t stride,
    size_t in_channels, size_t out_channels,
    size_t layerIndex,
    const PublicKey<DCRTPoly>& publicKey,
    const PrivateKey<DCRTPoly>& secretKey) {

    std::string layerPrefix = "conv" + std::to_string(layerIndex);
    auto filterPath = pathPrefix + "/" + layerPrefix + "_weight.txt";
    auto biasPath   = pathPrefix + "/" + layerPrefix + "_bias.txt";
    auto gammaPath  = pathPrefix + "/" + layerPrefix + "_bn_gamma.txt";
    auto betaPath   = pathPrefix + "/" + layerPrefix + "_bn_beta.txt";
    auto meanPath   = pathPrefix + "/" + layerPrefix + "_bn_mean.txt";
    auto varPath    = pathPrefix + "/" + layerPrefix + "_bn_var.txt";

    auto filters = LoadFromTxt(filterPath);
    auto biases  = LoadFromTxt(biasPath);
    auto gammas  = LoadFromTxt(gammaPath);
    auto betas   = LoadFromTxt(betaPath);
    auto means   = LoadFromTxt(meanPath);
    auto vars    = LoadFromTxt(varPath);

    std::vector<Ciphertext<DCRTPoly>> outputs;
    auto rotIndices = GenerateRotationIndices(filterH, filterW, inputW);
    cc->EvalRotateKeyGen(secretKey, rotIndices);

    for (size_t out_ch = 0; out_ch < out_channels; out_ch++) {
        Ciphertext<DCRTPoly> acc;
        for (size_t in_ch = 0; in_ch < in_channels; in_ch++) {
            size_t base = (out_ch * in_channels + in_ch) * filterH * filterW;
            std::vector<double> filter(filters.begin() + base, filters.begin() + base + filterH * filterW);
            auto ct = GeneralConv2D_CKKS(cc, ct_input_channels[in_ch], filter, 0.0, inputH, inputW, filterH, filterW, stride, rotIndices, publicKey);
            // ct = cc->Rescale(ct);
            acc = (in_ch == 0) ? ct : cc->EvalAdd(acc, ct);
        }
        acc = GeneralBatchNorm_CKKS(cc, acc, gammas[out_ch], betas[out_ch], means[out_ch], vars[out_ch], cc->GetEncodingParams()->GetBatchSize());
        // acc = cc->Rescale(acc);
        std::cout << "[Conv] Level: " << acc->GetLevel() << ", Scale: " << acc->GetScalingFactor() << std::endl;
        outputs.push_back(acc);
    }
    return outputs;
}

Ciphertext<DCRTPoly> ApproxReLU4(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    auto pt_half = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.5));
    auto pt_coeff2 = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.204875));
    auto pt_coeff4 = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, -0.0063896));
    auto pt_const = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.234606));

    // x1 = 0.5 * x
    auto x1 = cc->EvalMult(ct_x, pt_half);
    x1 = cc->Rescale(x1);

    // x2 = 0.204875 * x^2
    auto x2_raw = cc->EvalMult(ct_x, ct_x);
    x2_raw = cc->Rescale(x2_raw);
    auto x2 = cc->EvalMult(x2_raw, pt_coeff2);
    x2 = cc->Rescale(x2);

    // x4 = -0.0063896 * (x^2)^2
    auto x4_raw = cc->EvalMult(x2_raw, x2_raw);  // x^4
    x4_raw = cc->Rescale(x4_raw);
    auto x4 = cc->EvalMult(x4_raw, pt_coeff4);
    x4 = cc->Rescale(x4);

    // Match scale
    x1->SetScalingFactor(x4->GetScalingFactor());
    x2->SetScalingFactor(x4->GetScalingFactor());

    auto sum = cc->EvalAdd(x1, x2);
    sum = cc->EvalAdd(sum, x4);
    return cc->EvalAdd(sum, pt_const);  // pt_const는 scale 자동 정렬됨
}



std::vector<Ciphertext<DCRTPoly>> ApplyApproxReLU4_All(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels) {

    std::vector<Ciphertext<DCRTPoly>> activated;
    for (auto& ct : ct_channels) {
        std::cout << "[RELU] Level: " << ct->GetLevel() << ", Scale: " << ct->GetScalingFactor() << std::endl;
        activated.push_back(ApproxReLU4(cc, ct));
    }
    return activated;
}

std::vector<Ciphertext<DCRTPoly>> AvgPool2D_MultiChannel(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    size_t inputH, size_t inputW,
    const PublicKey<DCRTPoly>& pk,
    const std::vector<int>& rotIndices) {

    std::vector<Ciphertext<DCRTPoly>> pooled;
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    size_t outH = inputH / 2, outW = inputW / 2;

    std::vector<double> base_mask(slotCount, 0.0);
    base_mask[0] = 0.25;
    auto pt_base = cc->MakeCKKSPackedPlaintext(base_mask);

    for (auto& ct : ct_channels) {
        std::vector<Ciphertext<DCRTPoly>> rotated;
        for (size_t dy = 0; dy < 2; dy++) {
            for (size_t dx = 0; dx < 2; dx++) {
                int rot = dy * inputW + dx;
                auto rot_ct = cc->EvalRotate(ct, rot);
                auto masked = cc->EvalMult(rot_ct, pt_base);
                
                rotated.push_back(masked);
            }
        }
        auto ct_sum = cc->EvalAddMany(rotated);

        std::vector<double> mask(slotCount, 0.25);
        for (size_t i = 0; i < outH; i++) {
            for (size_t j = 0; j < outW; j++) {
                size_t idx = i * inputW + j;
                mask[idx] = 0.25;  // 실제 pooling 위치
            }
        }
        auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);
        auto pooled_ct = cc->EvalMult(ct_sum, pt_mask);
        pooled_ct = cc->Rescale(pooled_ct);  // ✅ 여기 추가
        pooled.push_back(pooled_ct);
    }
    return pooled;
} 
