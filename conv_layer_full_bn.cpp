#include "openfhe.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace lbcrypto;

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

Ciphertext<DCRTPoly> Conv2D_CKKS_SIMD_Masked(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::vector<double>& filter,
    size_t inputH, size_t inputW,
    size_t filterH, size_t filterW,
    size_t stride,
    const PublicKey<DCRTPoly>& pubkey
) {
    size_t outH = (inputH - filterH) / stride + 1;
    size_t outW = (inputW - filterW) / stride + 1;
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    std::vector<Ciphertext<DCRTPoly>> partials;

    for (size_t dy = 0; dy < filterH; dy++) {
        for (size_t dx = 0; dx < filterW; dx++) {
            size_t idx = dy * filterW + dx;
            int rotAmount = dy * inputW + dx;
            auto rotated = cc->EvalRotate(ct_input, rotAmount);

            std::vector<double> mask(slotCount, 0.0);
            for (size_t i = 0; i < outH; i++) {
                for (size_t j = 0; j < outW; j++) {
                    size_t baseRow = i * stride;
                    size_t baseCol = j * stride;
                    size_t inputIndex = (baseRow + dy) * inputW + (baseCol + dx);
                    size_t flatIdx = i * outW + j;
                    if (inputIndex < slotCount && flatIdx < slotCount)
                        mask[flatIdx] += filter[idx];
                }
            }
            auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);
            auto ct_partial = cc->EvalMult(rotated, pt_mask);
            partials.push_back(ct_partial);
        }
    }

    return cc->EvalAddMany(partials);

}

Ciphertext<DCRTPoly> BatchNormOnCiphertext(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    double gamma, double beta,
    double mean, double var,
    double bias,
    size_t slotCount
) {
    double epsilon = 1e-5;
    double a = gamma / std::sqrt(var + epsilon);
    double b = -a * mean + a * bias + beta;

    std::vector<double> a_vec(slotCount, a);
    std::vector<double> b_vec(slotCount, b);
    auto pt_a = cc->MakeCKKSPackedPlaintext(a_vec);
    auto pt_b = cc->MakeCKKSPackedPlaintext(b_vec);

    auto ct_scaled = cc->EvalMult(ct_input, pt_a);
    return cc->EvalAdd(ct_scaled, pt_b);
}

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();

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

    auto t_load_start = std::chrono::high_resolution_clock::now();

    auto img = LoadFromTxt("../input_image.txt");
    auto filters = LoadFromTxt("../lenet_weights_epoch(10)/conv1_weight.txt");
    auto biases = LoadFromTxt("../lenet_weights_epoch(10)/conv1_bias.txt");
    auto bn_gamma = LoadFromTxt("../lenet_weights_epoch(10)/conv1_bn_gamma.txt");
    auto bn_beta  = LoadFromTxt("../lenet_weights_epoch(10)/conv1_bn_beta.txt");
    auto bn_mean  = LoadFromTxt("../lenet_weights_epoch(10)/conv1_bn_mean.txt");
    auto bn_var   = LoadFromTxt("../lenet_weights_epoch(10)/conv1_bn_var.txt");

    auto t_load_end = std::chrono::high_resolution_clock::now();
    std::cout << "[Time] Data Load: " << std::chrono::duration<double>(t_load_end - t_load_start).count() << " sec\n";

    Plaintext pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);

    size_t stride = 1;
    size_t inputH = 32, inputW = 32;
    size_t filterH = 5, filterW = 5;
    size_t channel = 6;
    size_t outH = (inputH - filterH) / stride + 1;
    size_t outW = (inputW - filterW) / stride + 1;

    auto t_keygen_start = std::chrono::high_resolution_clock::now();

    std::set<int> rot_index_set;
    for (size_t dy = 0; dy < filterH; dy++) {
        for (size_t dx = 0; dx < filterW; dx++) {
            int rot = dy * inputW + dx;
            rot_index_set.insert(rot);
        }
    }
    std::vector<int> rot_indices(rot_index_set.begin(), rot_index_set.end());
    cc->EvalAtIndexKeyGen(keys.secretKey, rot_indices);

    auto t_keygen_end = std::chrono::high_resolution_clock::now();
    std::cout << "[Time] Rotation KeyGen: " << std::chrono::duration<double>(t_keygen_end - t_keygen_start).count() << " sec\n";

    for (size_t ch = 0; ch < channel; ch++) {
        std::vector<double> filter(filters.begin() + ch * 25, filters.begin() + (ch + 1) * 25);

        auto t_conv_start = std::chrono::high_resolution_clock::now();

        auto ct_conv = Conv2D_CKKS_SIMD_Masked(
            cc, ct_img, filter,
            inputH, inputW,
            filterH, filterW,
            stride,
            keys.publicKey
        );

        auto t_conv_end = std::chrono::high_resolution_clock::now();
        std::cout << "[Time] Conv2D (channel " << ch << "): " << std::chrono::duration<double>(t_conv_end - t_conv_start).count() << " sec\n";

        auto t_bn_start = std::chrono::high_resolution_clock::now();

        auto ct_bn = BatchNormOnCiphertext(
            cc, ct_conv,
            bn_gamma[ch], bn_beta[ch],
            bn_mean[ch], bn_var[ch],
            biases[ch],
            outH * outW
        );

        auto t_bn_end = std::chrono::high_resolution_clock::now();
        std::cout << "[Time] BatchNorm (channel " << ch << "): " << std::chrono::duration<double>(t_bn_end - t_bn_start).count() << " sec\n";

        Plaintext pt;
        cc->Decrypt(keys.secretKey, ct_bn, &pt);
        pt->SetLength(outH * outW);
        auto vec = pt->GetRealPackedValue();

        std::string filename = "conv1_output_channel_" + std::to_string(ch) + ".txt";
        std::ofstream outfile(filename);
        for (size_t i = 0; i < outH; i++) {
            for (size_t j = 0; j < outW; j++) {
                outfile << vec[i * outW + j];
                if (j != outW - 1) outfile << ",\n";
            }
            outfile << "\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "[Time] Total Elapsed: " << std::chrono::duration<double>(t_end - t_start).count() << " sec\n";

    return 0;
}