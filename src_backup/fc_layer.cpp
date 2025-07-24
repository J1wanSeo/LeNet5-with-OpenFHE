#include "openfhe.h"
#include "conv_bn_module.h"
#include <vector>
#include <iostream>

using namespace lbcrypto;

int main() {
    // ===============================
    // Step 0. CKKS context setup
    // ===============================
    uint32_t dcrtBits= 59;
    uint32_t ringDim = 1 << 16;
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    uint32_t batchSize = 16384;
    uint32_t multDepth = 3;
    lbcrypto::SecurityLevel securityLevel = lbcrypto::HEStd_NotSet;

    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(securityLevel);
    parameters.SetRingDim(ringDim);

    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
    cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    size_t fc_out_dim = 84;
    size_t fc_in_dim = 120;
    
    // Key generation
    KeyPair keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<int> rot_keys;
    for (size_t k = 1; k < fc_in_dim; k <<= 1) {
        rot_keys.push_back(k);
    }
    // 출력 슬롯 정렬을 위한 추가 회전 키들
    for (size_t i = 1; i < fc_out_dim; i++) {
        rot_keys.push_back(-i);
    }
    cc->EvalAtIndexKeyGen(keys.secretKey, rot_keys);


    std::cout << "CKKS context and keys generated." << std::endl;

    // ================================
    // Step 1. Example FC weight matrix (2x8)
    // ================================
    // FC 
    


    auto W = LoadFromTxt("../lenet_weights_epoch(10)/fc1_weight.txt");
    auto bias = LoadFromTxt("../lenet_weights_epoch(10)/fc1_bias.txt");   

    auto x = LoadFromTxt("../results/fc1_input.txt");
    Plaintext pt_x = cc->MakeCKKSPackedPlaintext(x);
    auto ct_x = cc->Encrypt(keys.publicKey, pt_x);

    // ===============================
    // Step 3. FC Layer: matrix-vector Multiplication
    // ===============================

    Ciphertext<DCRTPoly> ct_output;
    std::vector<Ciphertext<DCRTPoly>> rotated_sums;

    
    for (size_t i = 0; i < fc_out_dim; i++) {
        // i번째 출력 뉴런의 가중치 벡터
        std::vector<double> w_i(W.begin() + i * fc_in_dim,
                        W.begin() + (i + 1) * fc_in_dim);
        
        // 가중치를 슬롯 위치에 맞게 배치
        std::vector<double> w_i_positioned(batchSize, 0.0);
        for (size_t j = 0; j < fc_in_dim; j++) {
            w_i_positioned[j] = w_i[j];
        }
        
        Plaintext pt_wi = cc->MakeCKKSPackedPlaintext(w_i_positioned);
        auto ct_mult = cc->EvalMult(ct_x, pt_wi);

        // 내적을 위한 회전 및 누적 (로그 복잡도)
        for (size_t k = 1; k < fc_in_dim; k <<= 1) {
            auto rotated = cc->EvalRotate(ct_mult, k);
            ct_mult = cc->EvalAdd(ct_mult, rotated);
        }

        // Rotate result into proper slot (index i)
        auto ct_shifted = cc->EvalRotate(ct_mult, -static_cast<int>(i));
        rotated_sums.push_back(ct_shifted);
        }

        ct_output = cc->EvalAddMany(rotated_sums);

        auto pt_bias = cc->MakeCKKSPackedPlaintext(bias);
        ct_output = cc->EvalAdd(ct_output, pt_bias);
    }

 
    
    //////////////////////////////////////////
    // 4. Decrypt and print results
    //////////////////////////////////////////

    Plaintext pt;
    cc->Decrypt(keys.secretKey, ct_output, &pt);
    pt->SetLength(fc_out_dim);
    auto vec = pt->GetRealPackedValue();

    std::string filename = "../results/fc1_output.txt";
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(8);  

    for (size_t i = 0; i < fc_out_dim; i++) {
        out << vec[i];
        if (i < fc_out_dim - 1) out << ",\n";
    }
    std::cout << "FC result saved to " << filename << std::endl;

    return 0;

}
