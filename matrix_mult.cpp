#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

int main() {
    // ===============================
    // Step 0. CKKS context setup
    // ===============================
    uint32_t dcrtBits= 59;
    uint32_t ringDim = 1 << 8;
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    uint32_t batchSize = 4;
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
    cc->EvalAtIndexKeyGen(keys.secretKey, {1, -1, 2, -2, 3, -3}); // rotation keys

    std::cout << "CKKS context and keys generated." << std::endl;

    // ====================================
    // Step 1. Define Matrix and Encryption
    // ====================================

    size_t m = 4; // rows
    size_t n = 4; // cols


    std::vector<double> M1 = {
	    { 1.0,  2.0,  3.0,  4.0},
	    { 5.0,  6.0,  7.0,  8.0},
	    { 9.0, 10.0, 11.0, 12.0},
	    {13.0, 14.0, 15.0, 16.0}};

    std::vector<double> M2 = {
	    { 1.0,  2.0,  3.0,  4.0},
	    { 5.0,  6.0,  7.0,  8.0},
	    { 9.0, 10.0, 11.0, 12.0},
	    {13.0, 14.0, 15.0, 16.0}};
    
    std::vector<std::vector<double>> M1_cols(n, std::vector<double>(m));
    for (size_t j = 0; j < n; j++)
	    for (size_t i = 0; i < m; i++)
		    M1_cols[j][i] = M1[i][j];


    std::vector<std::vector<double>> M2_cols(n, std::vector<double>(m));
    for (size_t j = 0; j < n; j++)
	    for (size_t i = 0; i < m; i++)
		    M2_cols[j][i] = M2[i][j];

    std::vector<Ciphertext<DCRTPoly>> ctxt_M1_cols;
    for (const auto& col : M1_cols) {
	    Plaintext pt = cc->MakeCKKSPackedPlaintext(col);
	    auto ct = cc->Encrypt(keys.publicKey, pt);
	    ctxt_M1_cols.push_back(ct);
    }


    std::vector<Ciphertext<DCRTPoly>> ctxt_M2_cols;
    for (const auto& col : M2_cols) {
	    Plaintext pt = cc->MakeCKKSPackedPlaintext(col);
	    auto ct = cc->Encrypt(keys.publicKey, pt);
	    ctxt_M2_cols.push_back(ct);
    }

    // ===============================
    // Step 2. Matrix Replicate Function
    // ===============================

    



    // ===============================
    // Step 3. Matrix multiplication
    // ===============================
    std::vector<Ciphertext<DCRTPoly>> ct_result_rows;

    for (size_t i = 0; i < m; ++i) {
        // Multiply each row with input vector using EvalMult + rotations + EvalAdd
        Ciphertext<DCRTPoly> ct_row_sum;

        for (size_t j = 0; j < n; ++j) {
            // Rotate input vector by -j to align x_j to first slot
	    std::cout << "rotating by" << -static_cast<int>(j) << std::endl;
	    auto ct_rot = cc->EvalAtIndex(ct_x, -static_cast<int>(j));

            // Multiply rotated ciphertext with W[i][j]
            auto ct_mul = cc->EvalMult(ct_rot, W[i][j]);

            if (j == 0)
                ct_row_sum = ct_mul;
            else
                ct_row_sum = cc->EvalAdd(ct_row_sum, ct_mul);
        }

        ct_result_rows.push_back(ct_row_sum);
    }

    // ===============================
    // Step 4. Decrypt & print result
    // ===============================
    for (size_t i = 0; i < m; ++i) {
        Plaintext pt_res;
        cc->Decrypt(keys.secretKey, ct_result_rows[i], &pt_res);
        pt_res->SetLength(1);
        std::cout << "Row " << i << " result: " << pt_res << std::endl;
    }

    return 0;
}

