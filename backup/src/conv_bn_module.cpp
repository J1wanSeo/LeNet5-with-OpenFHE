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

    std::vector<double> bias_vector(slotCount, 0.0);
    std::vector<double> mask(slotCount, 0.0);
    
    for (size_t i = 0; i < outH; i++) {
        for (size_t j = 0; j < outW; j++) {
            size_t output_slot = (i * stride)*inputW + j*stride;
            if (output_slot < slotCount) {
                mask[output_slot] = 1.0;
                bias_vector[output_slot] = bias;
            }                    
        }
    }

    auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);

    for (size_t dy = 0; dy < filterH; dy++) {
        for (size_t dx = 0; dx < filterW; dx++) {
            size_t idx = dy * filterW + dx;
            int rotAmount = dy * inputW + dx;
            auto rotated = cc->EvalRotate(ct_input, rotAmount);           

            auto masked_rotated = cc->EvalMult(rotated, pt_mask);
            masked_rotated = cc->Rescale(masked_rotated);

            auto ct_weighted = cc->EvalMult(masked_rotated, filter[idx]);
            ct_weighted = cc->Rescale(ct_weighted);

            partials.push_back(ct_weighted);
        }
    }
    Ciphertext<DCRTPoly> result = cc->EvalAddMany(partials);
    
    auto pt_bias = cc->MakeCKKSPackedPlaintext(bias_vector);
    result = cc->EvalAdd(result, pt_bias);
    result = cc->EvalMult(result, pt_mask);

    return result;
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

    // size_t outH = (inputH - filterH) / stride + 1;
    // size_t outW = (inputW - filterW) / stride + 1;

    std::vector<Ciphertext<DCRTPoly>> outputs;
    auto rotIndices = GenerateRotationIndices(filterH, filterW, inputW);
    cc->EvalRotateKeyGen(secretKey, rotIndices);

    for (size_t out_ch = 0; out_ch < out_channels; out_ch++) {
        Ciphertext<DCRTPoly> ct_sum;
        for (size_t in_ch = 0; in_ch < in_channels; in_ch++) {
            size_t base = (out_ch * in_channels + in_ch) * filterH * filterW;
            double bias = biases[out_ch];
            std::vector<double> filter(filters.begin() + base, filters.begin() + base + filterH * filterW);

            auto ct = GeneralConv2D_CKKS(cc, ct_input_channels[in_ch], filter, bias, inputH, inputW, filterH, filterW, stride, rotIndices, publicKey);

            ct_sum = (!in_ch) ? ct : cc->EvalAdd(ct_sum, ct);

        }
        auto ct_bn = GeneralBatchNorm_CKKS(cc, ct_sum, gammas[out_ch], betas[out_ch], means[out_ch], vars[out_ch], cc->GetEncodingParams()->GetBatchSize());
        outputs.push_back(ct_bn);
    }
    return outputs;
}

std::vector<Ciphertext<DCRTPoly>> AvgPool2x2_MultiChannel_CKKS(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    size_t inputH, size_t inputW,
    const std::vector<int>& rotIndices) {

    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    size_t filterH = 2, filterW = 2, stride = 2;
    size_t outH = (inputH - filterH) / stride + 1;
    size_t outW = (inputW - filterW) / stride + 1;

    // stride 2 고려한 mask 생성 (모든 채널 공통 사용 가능)
    std::vector<double> mask(slotCount, 0.0);
    for (size_t i = 0; i < outH; i++) {
        for (size_t j = 0; j < outW; j++) {
            size_t output_slot = (i * stride) * inputW + (j * stride);
            if (output_slot < slotCount) {
                mask[output_slot] = 1.0;
            }
        }
    }
    auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);

    // 2x2 평균 filter weight = 1/4
    double weight = 0.25;

    // 채널별 AvgPool 수행
    std::vector<Ciphertext<DCRTPoly>> pooled;
    for (const auto& ct_input : ct_channels) {
        std::vector<Ciphertext<DCRTPoly>> partials;

        for (size_t dy = 0; dy < filterH; dy++) {
            for (size_t dx = 0; dx < filterW; dx++) {
                int rotAmount = dy * inputW + dx;
                auto rotated = cc->EvalRotate(ct_input, rotAmount);

                auto masked_rotated = cc->EvalMult(rotated, pt_mask);
                masked_rotated = cc->Rescale(masked_rotated);

                auto ct_weighted = cc->EvalMult(masked_rotated, weight);
                ct_weighted = cc->Rescale(ct_weighted);

                partials.push_back(ct_weighted);
            }
        }

        // partial sum → 최종 채널 avgpool 결과
        Ciphertext<DCRTPoly> result = cc->EvalAddMany(partials);
        pooled.push_back(result);
    }

    return pooled;
}

void SaveDecryptedConvOutput(
    CryptoContext<DCRTPoly> cc,
    const PrivateKey<DCRTPoly>& sk,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    size_t outH, size_t outW,
    const std::string& prefix) {

    for (size_t ch = 0; ch < ct_channels.size(); ++ch) {
        // 복호화
        Plaintext pt;
        cc->Decrypt(sk, ct_channels[ch], &pt);
        pt->SetLength(outH * outW);
        auto vec = pt->GetRealPackedValue();

        // 파일명 구성
        std::string filename = prefix + "_channel_" + std::to_string(ch) + ".txt";
        std::ofstream out(filename);
        out << std::fixed << std::setprecision(8);

        // 데이터 저장
        for (size_t i = 0; i < outH; i++) {
            for (size_t j = 0; j < outW; j++) {
                out << vec[i * outW + j];
                if (j < outW - 1) out << ",\n";
            }
            out << ",\n";
        }

        std::cout << "[INFO] Output saved: " << filename << std::endl;
    }
}

