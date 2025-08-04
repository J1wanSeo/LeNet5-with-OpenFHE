// conv_bn_module.h
#pragma once

#include "openfhe.h"
#include <vector>
#include <string>

using namespace lbcrypto;

std::vector<double> LoadFromTxt(const std::string& filename);
std::vector<int> GenerateRotationIndices(size_t filterH, size_t filterW, size_t inputW, size_t interleave);

// Conv2D
Ciphertext<DCRTPoly> GeneralConv2D_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::vector<double>& filter,
    double bias,
    size_t inputH, size_t inputW,
    size_t filterH, size_t filterW,
    size_t stride,
    size_t interleave,
    const PublicKey<DCRTPoly>& pk);

// BatchNorm
Ciphertext<DCRTPoly> GeneralBatchNorm_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    double gamma, double beta,
    double mean, double var,
    size_t slotCount);

// Conv+BN 전체 실행 (여러 채널 출력)
std::vector<Ciphertext<DCRTPoly>> ConvBnLayer(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_input_channels,
    const std::string& pathPrefix,
    size_t inputH, size_t inputW,
    size_t filterH, size_t filterW,
    size_t stride,
    size_t in_channels, size_t out_channels,
    size_t layerIndex,
    size_t interleave,
    const PublicKey<DCRTPoly>& publicKey,
    const PrivateKey<DCRTPoly>& secretKey);


std::vector<Ciphertext<DCRTPoly>> AvgPool2x2_MultiChannel_CKKS(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    size_t inputH, size_t inputW,
    size_t interleave);   

void SaveDecryptedConvOutput(
    CryptoContext<DCRTPoly> cc,
    const PrivateKey<DCRTPoly>& sk,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    size_t outH, size_t outW,
    const std::string& prefix);

std::vector<int> GenerateReAlignRotationKeys(
    int inputH, int inputW,
    int outputH, int outputW,
    int interleave
);

std::vector<Ciphertext<DCRTPoly>> ReAlignConvolutionResult_MultiChannel(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    int inputH, int inputW,
    int outputH, int outputW,
    int interleave
);

Ciphertext<DCRTPoly> ReAlignConvolutionResult(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    int inputH, int inputW,
    int outputH, int outputW,
    int interleave
);

std::vector<Ciphertext<DCRTPoly>> AvgPool2x2_MultiChannel_CKKS_SequentialPack(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    size_t inputH, size_t inputW);

std::vector<int> GenerateAvgPool2x2SequentialPackRotationIndices(
    size_t inputH, size_t inputW);