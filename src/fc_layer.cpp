#include "fc_layer.h"
#include "conv_bn_module.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>


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

    for (size_t i = 0; i < out_dim; i++) {
        std::vector<double> w_i(weights.begin() + i * in_dim, weights.begin() + (i + 1) * in_dim);

        auto pt_w = cc->MakeCKKSPackedPlaintext(w_i);
        auto ct_mult = cc->EvalMult(ct_input, pt_w);
        // ct_mult = cc->Rescale(ct_mult);

        // summation (내적, 계수 shift 방식)
        for (size_t k = 1; k < in_dim; k <<= 1) {
            auto rotated = cc->EvalRotate(ct_mult, k);
            ct_mult = cc->EvalAdd(ct_mult, rotated);
        }

        // bias 추가
        std::vector<double> bias_vec(cc->GetEncodingParams()->GetBatchSize(), bias[i]);
        auto pt_bias = cc->MakeCKKSPackedPlaintext(bias_vec);
        auto ct_neuron = cc->EvalAdd(ct_mult, pt_bias);

        auto ct_fc_bn = GeneralBatchNorm_CKKS(cc, ct_neuron, gammas[i], betas[i], means[i], vars[i], cc->GetEncodingParams()->GetBatchSize());

        // i번째 위치로 shift
        auto ct_shifted = cc->EvalRotate(ct_fc_bn, -(int)i);
        // ct_shifted = cc->Rescale(ct_shifted);   

        std::vector<double> mask(cc->GetEncodingParams()->GetBatchSize(), 0.0);
        mask[i] = 1.0;
        auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);
        ct_shifted = cc->EvalMult(ct_shifted, pt_mask);
        // ct_shifted = cc->Rescale(ct_shifted);

        ct_output = (i == 0) ? ct_shifted : cc->EvalAdd(ct_output, ct_shifted);
    }

    return ct_output;
}

// // 가중치 행렬을 대각선 벡터로 분해하는 함수 (in_dim x out_dim 행렬)
// std::vector<std::vector<double>> LoadWeightsToDiagonals(
//     const std::string& filepath,
//     size_t in_dim, size_t out_dim,
//     size_t slotCount) 
// {
//     // 1) 전체 가중치 로드 (row-major)
//     std::ifstream infile(filepath);
//     std::string content;
//     std::getline(infile, content);
//     std::stringstream ss(content);
//     std::string token;
//     std::vector<double> weights;
//     while (std::getline(ss, token, ',')) {
//         if (!token.empty()) weights.push_back(std::stod(token));
//     }
//     if (weights.size() != in_dim * out_dim) {
//         std::cerr << "[Error] Weights size mismatch\n";
//         exit(1);
//     }

//     // 2) 대각선 개수: in_dim + out_dim - 1
//     size_t num_diags = in_dim + out_dim - 1;
//     std::vector<std::vector<double>> diag_weights(num_diags);

//     // 3) 대각선별 길이: 대각선에 포함된 원소 수
//     // 대각선 인덱스 d: c - r + (in_dim - 1) or (out_dim - 1) - c + r (계산 방식에 따라 다름)
//     // 여기서는 out_dim 행, in_dim 열로 보고 c - r + (out_dim -1) 사용

//     for (size_t r = 0; r < out_dim; r++) {
//         for (size_t c = 0; c < in_dim; c++) {
//             size_t d = c - r + (out_dim - 1); // 대각선 인덱스
//             // 대각선별 위치는 vector 뒤에 push_back
//             diag_weights[d].push_back(weights[r * in_dim + c]);
//         }
//     }

//     // 4) 각 대각선 벡터를 slotCount 크기로 0-padding
//     for (auto& diag : diag_weights) {
//         while (diag.size() < slotCount) diag.push_back(0.0);
//     }

//     return diag_weights;
// }

// // 대각선 방식 FC 구현 함수
// Ciphertext<DCRTPoly> DiagonalFC_CKKS(
//     CryptoContext<DCRTPoly> cc,
//     const Ciphertext<DCRTPoly>& ct_input,
//     const std::string& pathPrefix,
//     size_t in_dim, size_t out_dim,
//     size_t layerIndex,
//     const PublicKey<DCRTPoly>& pk)
// {
//     std::string layerPrefix = "fc" + std::to_string(layerIndex);
//     auto filterPath = pathPrefix + "/" + layerPrefix + "_weight.txt";
//     auto biasPath   = pathPrefix + "/" + layerPrefix + "_bias.txt";
//     auto gammaPath  = pathPrefix + "/" + layerPrefix + "_bn_gamma.txt";
//     auto betaPath   = pathPrefix + "/" + layerPrefix + "_bn_beta.txt";
//     auto meanPath   = pathPrefix + "/" + layerPrefix + "_bn_mean.txt";
//     auto varPath    = pathPrefix + "/" + layerPrefix + "_bn_var.txt";

//     size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

//     // 1) 대각선 가중치 로드 및 분해
//     auto diag_weights = LoadWeightsToDiagonals(filterPath, in_dim, out_dim, slotCount);

//     // 2) bias, gamma, beta, mean, var 로드 (필요 시)
//     auto bias  = LoadFromTxt(biasPath);
//     auto gammas  = LoadFromTxt(gammaPath);
//     auto betas   = LoadFromTxt(betaPath);
//     auto means   = LoadFromTxt(meanPath);
//     auto vars    = LoadFromTxt(varPath);

//     size_t num_diags = diag_weights.size();

//     std::vector<Ciphertext<DCRTPoly>> partials(num_diags);

//     for (size_t d = 0; d < num_diags; d++) {
//         // 3) 입력 ciphertext 회전
//         auto rotated = (d == (out_dim - 1)) ? ct_input : cc->EvalRotate(ct_input, (int)(d - (out_dim - 1)));

//         // 4) 평문 가중치 생성
//         auto pt_w = cc->MakeCKKSPackedPlaintext(diag_weights[d]);

//         // 5) 원소별 곱 및 스케일 조정
//         auto mult = cc->EvalMult(rotated, pt_w);
//         mult = cc->Rescale(mult);

//         partials[d] = mult;
//     }

//     // 6) 대각선별 결과 모두 더하기
//     auto ct_sum = cc->EvalAddMany(partials);

//     // 7) Bias 및 BatchNorm 적용 (출력 벡터 위치 그대로 유지됨)
//     // bias 등은 slotCount 길이 벡터로 만들고 추가
//     auto pt_bias = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.0));
//     for (size_t i = 0; i < out_dim; i++) {
//         pt_bias->SetValue(i, bias[i]);
//     }
//     auto ct_bias_added = cc->EvalAdd(ct_sum, pt_bias);

//     // BatchNorm: 채널 단위가 아니므로 out_dim 길이 기준으로 적용 (단순화)
//     // 여기서는 BatchNorm을 간단히 적용 (벡터 원소별 다르게 적용하려면 추가 코드 필요)
//     // 보통 BatchNorm은 합산 후 다항식 근사 등으로 처리 가능

//     // 예: 여기선 생략하거나 필요 시 구현

//     return ct_bias_added;
// }

void SaveDecryptedFCOutput(
    CryptoContext<DCRTPoly> cc,
    const PrivateKey<DCRTPoly>& sk,
    const Ciphertext<DCRTPoly>& ct_output,
    size_t out_dim,
    const std::string& filename) {

    Plaintext pt;
    cc->Decrypt(sk, ct_output, &pt);
    pt->SetLength(out_dim);
    auto vec = pt->GetRealPackedValue();

    std::ofstream out(filename);
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

