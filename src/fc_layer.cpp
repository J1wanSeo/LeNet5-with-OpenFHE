#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

std::vector<std::vector<double>> RowMajorToDiagonal(
    const std::vector<double>& weight,
    size_t fc_out_dim,
    size_t fc_in_dim)
{
    std::vector<std::vector<double>> diagonals(fc_in_dim, std::vector<double>(fc_out_dim, 0.0));
    for (size_t k = 0; k < fc_in_dim; k++) {
        for (size_t i = 0; i < fc_out_dim; i++) {
            size_t slotIdx = (i + k) % fc_out_dim;
            diagonals[k][slotIdx] = weight[i * fc_in_dim + k];
        }
    }
    return diagonals;
}


int main() {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetRingDim(1 << 14);
    params.SetScalingModSize(40);
    params.SetBatchSize(4096);
    params.SetMultiplicativeDepth(5);
    params.SetScalingTechnique(FLEXIBLEAUTO);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // Rotation key: input_dim만큼 생성
    size_t fc_in_dim = 20, fc_out_dim = 5;
    std::vector<int> rotKeys;
    for (size_t k = 1; k < fc_in_dim; ++k)
        rotKeys.push_back(k);
    cc->EvalAtIndexKeyGen(keys.secretKey, rotKeys);

    // 입력 (1~20)
    std::vector<double> x(fc_in_dim);
    for (size_t i = 0; i < fc_in_dim; i++)
        x[i] = i + 1;

    auto pt_x = cc->MakeCKKSPackedPlaintext(x);
    auto ct_x = cc->Encrypt(keys.publicKey, pt_x);

    // Weight, bias (테스트용)
    std::vector<double> weights(fc_out_dim * fc_in_dim);
    std::vector<double> bias(fc_out_dim, 0.0);
    for (size_t i = 0; i < fc_out_dim * fc_in_dim; i++)
        weights[i] = i * 0.1;

    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    Ciphertext<DCRTPoly> ct_output;

    for (size_t k = 0; k < fc_in_dim; k++) {
        auto ct_rot = (k == 0) ? ct_x : cc->EvalRotate(ct_x, k);
        std::vector<double> diag(slotCount, 0.0);
        for (size_t i = 0; i < fc_out_dim; i++)
            diag[(i + k) % slotCount] = weights[i * fc_in_dim + k];
    
        auto pt_diag = cc->MakeCKKSPackedPlaintext(diag);
        auto ct_mul = cc->EvalMult(ct_rot, pt_diag);
        cc->ModReduceInPlace(ct_mul);
    
        if (!ct_output)
            ct_output = ct_mul;
        else {
            while (ct_output->GetLevel() > ct_mul->GetLevel())
                cc->ModReduceInPlace(ct_output);
            while (ct_output->GetLevel() < ct_mul->GetLevel())
                cc->ModReduceInPlace(ct_mul);
            ct_output = cc->EvalAdd(ct_output, ct_mul);
        }
    }
    
    // bias는 slot 0~4만 (나머지 0)
    std::vector<double> bias_vec(slotCount, 0.0);
    for (size_t i = 0; i < fc_out_dim; i++)
        bias_vec[i] = bias[i];
    
    auto pt_bias = cc->MakeCKKSPackedPlaintext(bias_vec);
    pt_bias->SetScalingFactor(ct_output->GetScalingFactor());
    ct_output = cc->EvalAdd(ct_output, pt_bias);




    // 복호화 및 출력
    lbcrypto::Plaintext pt;
    cc->Decrypt(keys.secretKey, ct_output, &pt);
    pt->SetLength(fc_out_dim);

    auto result = pt->GetRealPackedValue();
    std::cout << "[Diagonal FC output]:" << std::endl;
    for (size_t i = 0; i < fc_out_dim; i++)
        std::cout << "y[" << i << "] = " << result[i] << std::endl;
    return 0;
}
