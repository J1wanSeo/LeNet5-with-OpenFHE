#include "openfhe.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace lbcrypto;


// Load Data from txt

std::vector<double> LoadFromTxt(const std::string& filename) {
    std::ifstream infile(filename);
    std::vector<double> data;
    std::string content;
    std::getline(infile, content); 

    std::stringstream ss(content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            try {
                data.push_back(std::stod(token));
            } catch (const std::exception& e) {
                std::cerr << "Failed: '" << token << "' (" << e.what() << ")" << std::endl;
            }
        }
    }

    return data;
}


Ciphertext<DCRTPoly> Conv2D_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::vector<double>& filter,
    size_t inputH,
    size_t inputW,
    size_t filterH,
    size_t filterW,
    double bias,
    const PublicKey<DCRTPoly>& pubkey,
    const std::vector<int>& rot_indices,
    size_t baseRow, size_t baseCol,
    std::unordered_map<int, Ciphertext<DCRTPoly>>& rotation_cache
) {

    std::vector<Ciphertext<DCRTPoly>> rotatedCts;

    for (size_t dy = 0; dy < filterH; dy++) {
        for (size_t dx = 0; dx < filterW; dx++) {
            size_t idx = dy * filterW + dx;
            int rotAmount = (baseRow + dy) * inputW + (baseCol + dx);
            
	    Ciphertext<DCRTPoly> rotated;
            if (rotation_cache.find(rotAmount) == rotation_cache.end()) {
                rotated = cc->EvalRotate(ct_input, rotAmount);
                rotation_cache[rotAmount] = rotated; // ⬅️ 캐싱
            } else {
                rotated = rotation_cache[rotAmount];
            }
	    
            std::vector<double> wt(inputH * inputW, 0.0);
            wt[0] = filter[idx];
            auto pt_w = cc->MakeCKKSPackedPlaintext(wt);

            auto ct_mul = cc->EvalMult(rotated, pt_w);
            rotatedCts.push_back(ct_mul);
        }
    }

    auto ct_sum = cc->EvalAddMany(rotatedCts);

    std::vector<double> biasVec(inputH * inputW, 0.0);
    biasVec[0] = bias;
    auto pt_bias = cc->MakeCKKSPackedPlaintext(biasVec);

    return cc->EvalAdd(ct_sum, pt_bias);
}

std::vector<double> Conv2D_FullOutput_Stride(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::vector<double>& filter,
    size_t inputH, size_t inputW,
    size_t filterH, size_t filterW,
    size_t stride,
    double bias,
    const PublicKey<DCRTPoly>& pubkey,
    const std::vector<int>& rot_indices,
    const PrivateKey<DCRTPoly>& secretKey
) {
//    size_t outH = (inputH - filterH) / stride + 1;
//    size_t outW = (inputW - filterW) / stride + 1;

    std::unordered_map<int, Ciphertext<DCRTPoly>> rotation_cache;
    std::vector<double> outputs;

    for (size_t i = 0; i <= inputH - filterH; i += stride) {
        for (size_t j = 0; j <= inputW - filterW; j += stride) {
            auto ct_out = Conv2D_CKKS(cc, ct_input, filter, inputH, inputW,
                                      filterH, filterW, bias, pubkey, rot_indices,
                                      i, j, rotation_cache);

            Plaintext pt_out;
            cc->Decrypt(secretKey, ct_out, &pt_out);
            pt_out->SetLength(1);
            outputs.push_back(pt_out->GetRealPackedValue()[0]);
        }
    }

    return outputs;
}

Ciphertext<DCRTPoly> Conv2D_FullOutputCipher_Stride(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::vector<double>& filter,
    size_t inputH, size_t inputW,
    size_t filterH, size_t filterW,
    size_t stride,
    double bias,
    const PublicKey<DCRTPoly>& pubkey,
    const std::vector<int>& rot_indices
) {
    //size_t outH = (inputH - filterH) / stride + 1;
    size_t outW = (inputW - filterW) / stride + 1;
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    std::unordered_map<int, Ciphertext<DCRTPoly>> rotation_cache;

    std::vector<Ciphertext<DCRTPoly>> ct_outputs;

    for (size_t i = 0; i <= inputH - filterH; i += stride) {
        for (size_t j = 0; j <= inputW - filterW; j += stride) {
            auto ct_out = Conv2D_CKKS(cc, ct_input, filter, inputH, inputW,
                                      filterH, filterW, bias, pubkey, rot_indices,
                                      i, j, rotation_cache);

            // 위치 (i, j) → flat index
            size_t flatIdx = (i / stride) * outW + (j / stride);

            // 이 위치에 넣기 위한 mask vector
            std::vector<double> mask(slotCount, 0.0);
            mask[flatIdx] = 1.0;
            auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);

            auto ct_masked = cc->EvalMult(ct_out, pt_mask);
            ct_outputs.push_back(ct_masked);
        }
    }

    return cc->EvalAddMany(ct_outputs);
}



Ciphertext<DCRTPoly> BatchNormOnCiphertext(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    double gamma, double beta,
    double mean, double var,
    size_t slotCount
) {
    double epsilon = 1e-5;
    double a = gamma / std::sqrt(var + epsilon);
    double b = -a * mean + beta;

    std::vector<double> a_vec(slotCount, a);
    std::vector<double> b_vec(slotCount, b);
    auto pt_a = cc->MakeCKKSPackedPlaintext(a_vec);
    auto pt_b = cc->MakeCKKSPackedPlaintext(b_vec);

    auto ct_scaled = cc->EvalMult(ct_input, pt_a);
    auto ct_bn = cc->EvalAdd(ct_scaled, pt_b);

    return ct_bn;
}


int main() {
    // CKKS Configuration
    CCParams<CryptoContextCKKSRNS> params;
    params.SetRingDim(1 << 14);
    params.SetScalingModSize(40);
    params.SetBatchSize(4096);
    params.SetMultiplicativeDepth(3);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    auto load_start = std::chrono::high_resolution_clock::now();
    // Load data from txt
    auto img = LoadFromTxt("../input_image.txt");
    auto filters = LoadFromTxt("../conv1_weight.txt");
    auto biases = LoadFromTxt("../conv1_bias.txt");
    auto bn_gamma = LoadFromTxt("../conv1_bn_gamma.txt");
    auto bn_beta  = LoadFromTxt("../conv1_bn_beta.txt");
    auto bn_mean  = LoadFromTxt("../conv1_bn_mean.txt");
    auto bn_var   = LoadFromTxt("../conv1_bn_var.txt");

    auto load_end = std::chrono::high_resolution_clock::now();
    std::cout << "Load Elapsed time (sec): " << std::chrono::duration<double>(load_end - load_start).count() << std::endl;
    Plaintext pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);

    size_t stride = 1;
    size_t inputH = 32, inputW = 32;
    size_t filterH = 5, filterW = 5;
    size_t channel = 6;
    size_t outH = (inputH - filterH) / stride + 1;
    size_t outW = (inputW - filterW) / stride + 1;

    std::set<int> rot_index_set;


    auto start = std::chrono::high_resolution_clock::now();

    for (size_t baseRow = 0; baseRow <= inputH - filterH; baseRow += stride) {
        for (size_t baseCol = 0; baseCol <= inputW - filterW; baseCol += stride) {
            for (size_t dy = 0; dy < filterH; dy++) {
                for (size_t dx = 0; dx < filterW; dx++) {
                    int rot = (baseRow + dy) * inputW + (baseCol + dx);
                    rot_index_set.insert(rot);
                }
            }
        }
    }

    std::vector<int> rot_indices(rot_index_set.begin(), rot_index_set.end());
    cc->EvalAtIndexKeyGen(keys.secretKey, rot_indices);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "rotation keygen Elapsed time (sec): " << std::chrono::duration<double>(end - start).count() << std::endl;

    for (size_t ch = 0; ch < channel; ch++) {
        std::vector<double> filter(filters.begin() + ch * 25, filters.begin() + (ch + 1) * 25);

	double bias = biases[ch];

	auto start = std::chrono::high_resolution_clock::now();
        auto ct_conv = Conv2D_FullOutputCipher_Stride(
            cc, ct_img, filter,
            inputH, inputW,
            filterH, filterW,
            stride,
            bias,
            keys.publicKey,
            rot_indices
           // keys.secretKey
        );
	
        auto ct_bn = BatchNormOnCiphertext(
            cc, ct_conv,
            bn_gamma[ch], bn_beta[ch],
            bn_mean[ch], bn_var[ch],
            inputH * inputW
        );

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Elapsed time (sec): " << std::chrono::duration<double>(end - start).count() << std::endl;


        Plaintext pt;
        cc->Decrypt(keys.secretKey, ct_bn, &pt);
        pt->SetLength(outH * outW);
        auto vec = pt->GetRealPackedValue();

        std::string filename = "conv1_output_channel_" + std::to_string(ch) + ".txt";
        std::ofstream outfile(filename);
        for (size_t i = 0; i < outH; i++) {
            for (size_t j = 0; j < outW; j++) {
                outfile << vec[i * outW + j];
                if (j != outW - 1) outfile << ",";
            }
            outfile << "\n";
    
        }   
    }
    return 0;
}
