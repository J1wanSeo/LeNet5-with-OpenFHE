#pragma once
#include "openfhe.h"
#include <vector>
#include <string>

using namespace lbcrypto;

Ciphertext<DCRTPoly> GeneralFC_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::string& pathPrefix,
    size_t in_dim, size_t out_dim,
    size_t layerIndex,
    const PublicKey<DCRTPoly>& pk);

Ciphertext<DCRTPoly> GeneralFC_wo_BN_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::string& pathPrefix,
    size_t in_dim, size_t out_dim,
    size_t layerIndex,
    const PublicKey<DCRTPoly>& pk);

void SaveDecryptedFCOutput(
    CryptoContext<DCRTPoly> cc,
    const PrivateKey<DCRTPoly>& sk,
    const Ciphertext<DCRTPoly>& ct_output,
    size_t out_dim,
    const std::string& filename);

