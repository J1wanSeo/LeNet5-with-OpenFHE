#include "openfhe.h"
#include <vector>
#include <iostream>

using namespace lbcrypto;

int main() {
    // ===============================
    // Step 0. CKKS context setup
    // ===============================
    uint32_t dcrtBits= 59;
    uint32_t ringDim = 1 << 10;
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    uint32_t batchSize = 512;
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

    // Key generation
    KeyPair keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<int32_t> rotation_indices;
    for (size_t i = 1; i < batchSize; i <<= 1) {
	    rotation_indices.push_back(i);
    }
    cc->EvalAtIndexKeyGen(keys.secretKey, rotation_indices); // rotation keys

    std::cout << "CKKS context and keys generated." << std::endl;

    // ================================
    // Step 1. Example FC weight matrix (2x8)
    // ================================
    // FC 
    
    size_t fc_out_dim = 120;
    size_t fc_in_dim = 400;

    // Creating Dummy weights: W[i][j] = 0.001 * (i+j)
    std::vector<std::vector<double>> W(fc_out_dim, std::vector<double>(fc_in_dim));
    for (size_t i = 0; i < fc_out_dim; i++) {
	    for(size_t j = 0; j < fc_in_dim; j++) {
		    W[i][j] = 0.001 * (i + j);
	    }
    }

    // Bias vector: b[i] = 0.1 * i
    std::vector<double> bias(fc_out_dim);
    for (size_t i = 0; i < fc_out_dim; i++) {
	    bias[i] = 0.1 * i;
    }

    // ================================
    // Step 2. Encrypt Input Vector x (size = 400) .. flatten output
    // ===============================
    std::vector<double> x(fc_in_dim, 1.0);
    Plaintext pt_x = cc->MakeCKKSPackedPlaintext(x);
    auto ct_x = cc->Encrypt(keys.publicKey, pt_x);

    // ===============================
    // Step 3. FC Layer: matrix-vector Multiplication
    // ===============================

    std::vector<Ciphertext<DCRTPoly>> ct_output;

    for (size_t i = 0; i < fc_out_dim; i++) {
	    Plaintext pt_wi = cc->MakeCKKSPackedPlaintext(W[i]);
	    auto ct_mult = cc->EvalMult(ct_x, pt_wi);

	    for(size_t k = 1; k < fc_in_dim; k <<= 1) {
		    auto rotated = cc->EvalRotate(ct_mult, k);
		    ct_mult = cc->EvalAdd(ct_mult, rotated);
            
	    }	

    std::vector<double> bias_vec(batchSize, bias[i]);
    Plaintext pt_bias = cc->MakeCKKSPackedPlaintext(bias_vec);
    auto ct_result = cc->EvalAdd(ct_mult, pt_bias);
        ct_output.push_back(ct_result);
    }

    //////////////////////////////////////////
    // 4. Decrypt and print results
    //////////////////////////////////////////
    for (size_t i = 0; i < ct_output.size(); i++) {
        Plaintext result;
        cc->Decrypt(keys.secretKey, ct_output[i], &result);
        result->SetLength(1); // inner product result in slot 0
        std::cout << "FC output[" << i << "] = " << result << std::endl;
    }

    return 0;

}
