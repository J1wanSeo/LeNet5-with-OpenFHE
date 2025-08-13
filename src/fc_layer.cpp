#include "fc_layer.h"
#include "conv_bn_module.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <omp.h>


// 일반적인 FC Layer (한 번에 out_dim 크기까지 합침)
Ciphertext<DCRTPoly> GeneralFC_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::string& pathPrefix,
    size_t in_dim, size_t out_dim,
    size_t layerIndex,
    const PublicKey<DCRTPoly>& pk) {

    Ciphertext<DCRTPoly> ct_output;

    std::string layerPrefix = "fc" + std::to_string(layerIndex);
    auto filterPath = pathPrefix + "/" + layerPrefix + "_weight.txt";
    auto biasPath   = pathPrefix + "/" + layerPrefix + "_bias.txt";
    auto gammaPath  = pathPrefix + "/" + layerPrefix + "_bn_gamma.txt";
    auto betaPath   = pathPrefix + "/" + layerPrefix + "_bn_beta.txt";
    auto meanPath   = pathPrefix + "/" + layerPrefix + "_bn_mean.txt";
    auto varPath    = pathPrefix + "/" + layerPrefix + "_bn_var.txt";

    auto weights = LoadFromTxt(filterPath);
    auto bias  = LoadFromTxt(biasPath);
    auto gammas  = LoadFromTxt(gammaPath);
    auto betas   = LoadFromTxt(betaPath);
    auto means   = LoadFromTxt(meanPath);
    auto vars    = LoadFromTxt(varPath);

    std::vector<Ciphertext<DCRTPoly>> partial_outputs(out_dim);

    #pragma omp parallel for
    for (size_t i = 0; i < out_dim; i++) {
        std::vector<double> w_i(weights.begin() + i * in_dim, weights.begin() + (i + 1) * in_dim);

        auto pt_w = cc->MakeCKKSPackedPlaintext(w_i);
        // auto ct_mult = cc->EvalMult(ct_input, pt_w);

        // // summation (내적, 계수 shift 방식)
        // for (size_t k = 1; k < in_dim; k <<= 1) {
        //     auto rotated = cc->EvalRotate(ct_mult, k);
        //     ct_mult = cc->EvalAdd(ct_mult, rotated);
        // }

        auto ct_mult = cc->EvalInnerProduct(ct_input, pt_w, in_dim);

        // bias 추가
        std::vector<double> bias_vec(cc->GetEncodingParams()->GetBatchSize(), bias[i]);
        auto pt_bias = cc->MakeCKKSPackedPlaintext(bias_vec);
        auto ct_neuron = cc->EvalAdd(ct_mult, pt_bias);

        auto ct_fc_bn = GeneralBatchNorm_CKKS(cc, ct_mult, gammas[i], betas[i], means[i], vars[i], cc->GetEncodingParams()->GetBatchSize());

        // i번째 위치로 shift
        auto ct_shifted = RotateByIndex(cc, ct_fc_bn, -(int)i, cc->GetEncodingParams()->GetBatchSize()); //->EvalRotate(ct_fc_bn, -(int)i);
        // ct_shifted = cc->Rescale(ct_shifted);   

        std::vector<double> mask(cc->GetEncodingParams()->GetBatchSize(), 0.0);
        mask[i] = 1.0;
        auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);
        ct_shifted = cc->EvalMult(ct_shifted, pt_mask);
        ct_shifted = cc->Rescale(ct_shifted);

        partial_outputs[i] = ct_shifted;
        
    }

    for (size_t i = 0; i < out_dim; i++) {
        if (i == 0) {
            ct_output = partial_outputs[i];
        } else {
            ct_output = cc->EvalAdd(ct_output, partial_outputs[i]);
        }
    }

    return ct_output;
}
// without BN
Ciphertext<DCRTPoly> GeneralFC_wo_BN_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::string& pathPrefix,
    size_t in_dim, size_t out_dim,
    size_t layerIndex,
    const PublicKey<DCRTPoly>& pk) {

    Ciphertext<DCRTPoly> ct_output;

    std::string layerPrefix = "fc" + std::to_string(layerIndex);
    auto filterPath = pathPrefix + "/" + layerPrefix + "_weight.txt";
    auto biasPath   = pathPrefix + "/" + layerPrefix + "_bias.txt";

    auto weights = LoadFromTxt(filterPath);
    auto bias  = LoadFromTxt(biasPath);

    std::vector<Ciphertext<DCRTPoly>> partial_outputs(out_dim);

    #pragma omp parallel for
    for (size_t i = 0; i < out_dim; i++) {
        std::vector<double> w_i(weights.begin() + i * in_dim, weights.begin() + (i + 1) * in_dim);

        auto pt_w = cc->MakeCKKSPackedPlaintext(w_i);
        // auto ct_mult = cc->EvalMult(ct_input, pt_w);

        // // summation (내적, 계수 shift 방식)
        // for (size_t k = 1; k < in_dim; k <<= 1) {
        //     auto rotated = cc->EvalRotate(ct_mult, k);
        //     ct_mult = cc->EvalAdd(ct_mult, rotated);
        // }

        auto ct_mult = cc->EvalInnerProduct(ct_input, pt_w, in_dim);

        // // bias 추가
        std::vector<double> bias_vec(cc->GetEncodingParams()->GetBatchSize(), bias[i]);
        auto pt_bias = cc->MakeCKKSPackedPlaintext(bias_vec);
        auto ct_neuron = cc->EvalAdd(ct_mult, pt_bias);

        // i번째 위치로 shift
        auto ct_shifted = RotateByIndex(cc, ct_mult, -(int)i, cc->GetEncodingParams()->GetBatchSize()); //cc->EvalRotate(ct_neuron, -(int)i);
        // ct_shifted = cc->Rescale(ct_shifted);   

        std::vector<double> mask(cc->GetEncodingParams()->GetBatchSize(), 0.0);
        mask[i] = 1.0;
        auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);
        ct_shifted = cc->EvalMult(ct_shifted, pt_mask);
        ct_shifted = cc->Rescale(ct_shifted);

        partial_outputs[i] = ct_shifted;
    }

    for (size_t i = 0; i < out_dim; i++) {
        if (i == 0) {
            ct_output = partial_outputs[i];
        } else {
            ct_output = cc->EvalAdd(ct_output, partial_outputs[i]);
        }
    }

    return ct_output;
}

void SaveDecryptedFCOutput(
    CryptoContext<DCRTPoly> cc,
    const PrivateKey<DCRTPoly>& sk,
    const Ciphertext<DCRTPoly>& ct_output,
    size_t out_dim,
    const std::string& filename) {

    Plaintext pt;

    cc->Decrypt(sk, ct_output, &pt);
    // pt->SetLength(out_dim);
    auto vec = pt->GetRealPackedValue();
    std::string filename_out = filename +  ".txt";
    std::ofstream out(filename_out);
    out << std::fixed << std::setprecision(8);

    for (size_t i = 0; i < out_dim; i++) {
        out << vec[i];
        if (i < out_dim - 1) out << ",\n";
    }
    out.close();
    std::cout << "[INFO] FC output saved: " << filename << std::endl;
}


// int main() {
//     // ... context/키 생성 생략

//     CCParams<CryptoContextCKKSRNS> params;
//     params.SetRingDim(1 << 16);
//     params.SetScalingModSize(40);
//     params.SetBatchSize(4096);
//     params.SetMultiplicativeDepth(20);
//     params.SetScalingTechnique(FLEXIBLEAUTO);

//     CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
//     cc->Enable(PKE);
//     cc->Enable(LEVELEDSHE);
//     cc->Enable(ADVANCEDSHE);

//     auto keys = cc->KeyGen();
//     cc->EvalMultKeyGen(keys.secretKey);

//     std::string path = "../lenet_weights_epoch(10)";

//     size_t fc_in_dim = 120;
//     size_t fc_out_dim = 84;

//     auto x = LoadFromTxt("../results/fc1_input.txt");
//     Plaintext pt_x = cc->MakeCKKSPackedPlaintext(x);
//     auto ct_x = cc->Encrypt(keys.publicKey, pt_x);

//     // Rotation key는 in_dim, out_dim에 맞게 미리 셋업 (conv에서처럼 따로 빼도 무방)
//     std::vector<int> rotIndices;
//     for (size_t k = 1; k < fc_in_dim; k <<= 1) rotIndices.push_back(k);
//     for (size_t i = 0; i < fc_out_dim; i++) rotIndices.push_back(-i);
//     cc->EvalAtIndexKeyGen(keys.secretKey, rotIndices);

//     auto ct_output = GeneralFC_CKKS(cc, ct_x, path, 120, 84, 1, keys.publicKey);

//     SaveDecryptedFCOutput(cc, keys.secretKey, ct_output, 84, "fc1_output.txt");
// }

