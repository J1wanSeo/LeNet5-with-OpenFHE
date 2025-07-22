#include "openfhe.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <set>

using namespace lbcrypto;

// 텍스트 파일에서 double 벡터를 로드하는 함수
std::vector<double> LoadFromTxt(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }
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
                std::cerr << "Failed to parse token: '" << token << "' (" << e.what() << ")" << std::endl;
            }
        }
    }
    return data;
}


Ciphertext<DCRTPoly> Conv2D_CKKS_SIMD_Masked(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    const std::vector<double>& filter,
    double bias,
    size_t inputH, size_t inputW,
    size_t filterH, size_t filterW,
    size_t stride) {
        
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
                    size_t padded_idx = i * inputW + j;
                    if (padded_idx < slotCount) {
                        mask[padded_idx] = 1.0;
                    }
                }
            }
            auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);
            auto masked_rotated = cc->EvalMult(rotated, pt_mask);
            auto ct_weighted = cc->EvalMult(masked_rotated, filter[idx]);

            partials.push_back(ct_weighted);
        }
    }
    Ciphertext<DCRTPoly> result = cc->EvalAddMany(partials);

    // ADDED: Add bias to the result
    // The bias is added to every relevant output slot.
    std::vector<double> bias_vector(slotCount, 0.0);
    for (size_t i = 0; i < outH; ++i) {
        for (size_t j = 0; j < outW; ++j) {
            size_t padded_idx = i * inputW + j;
            if (padded_idx < slotCount) {
                bias_vector[padded_idx] = bias;
            }
        }
    }
    auto pt_bias = cc->MakeCKKSPackedPlaintext(bias_vector);
    result = cc->EvalAdd(result, pt_bias);

    return result;
}


/**
 * @brief 암호문에 대해 배치 정규화를 수행합니다.
 * PyTorch의 `BatchNorm`과 동일한 결과를 내도록 수식을 수정했습니다.
 */
Ciphertext<DCRTPoly> BatchNormOnCiphertext(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    double gamma, double beta,
    double mean, double var,
    size_t slotCount
) {
    double epsilon = 1e-5;
    double a = gamma / std::sqrt(var + epsilon);
    double b = beta - a * mean;

    auto pt_a = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, a));
    auto pt_b = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, b));

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
    
    // 데이터 파일 경로는 실제 환경에 맞게 수정해주세요.
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
            rot_index_set.insert(dy * inputW + dx);
        }
    }
    cc->EvalRotateKeyGen(keys.secretKey, std::vector<int>(rot_index_set.begin(), rot_index_set.end()));

    auto t_keygen_end = std::chrono::high_resolution_clock::now();
    std::cout << "[Time] Rotation KeyGen: " << std::chrono::duration<double>(t_keygen_end - t_keygen_start).count() << " sec\n";

    for (size_t ch = 0; ch < channel; ch++) {
        std::vector<double> filter(filters.begin() + ch * 25, filters.begin() + (ch + 1) * 25);
        double bias = biases[ch];
        auto t_conv_start = std::chrono::high_resolution_clock::now();
        auto ct_conv = Conv2D_CKKS_SIMD_Masked(
            cc, ct_img, filter, bias,
            inputH, inputW, filterH, filterW, stride
        );
        auto t_conv_end = std::chrono::high_resolution_clock::now();
        std::cout << "[Time] Conv2D (channel " << ch << "): " << std::chrono::duration<double>(t_conv_end - t_conv_start).count() << " sec\n";

        // ✨ ADDITION: 배치 정규화 이전 결과(컨볼루션 출력) 저장
        Plaintext pt_conv_out;
        cc->Decrypt(keys.secretKey, ct_conv, &pt_conv_out);
        auto vec_conv = pt_conv_out->GetRealPackedValue();
        std::string conv_filename = "conv1_output_channel_b4_bn_" + std::to_string(ch) + ".txt";
        std::ofstream conv_outfile(conv_filename);
        conv_outfile << std::fixed << std::setprecision(8);
        for (size_t i = 0; i < outH; i++) {
            for (size_t j = 0; j < outW; j++) {
                conv_outfile << vec_conv[i * inputW + j];
                if (j < outW - 1) conv_outfile << ",\n";
            }
            if (i < outH - 1) conv_outfile << ",\n";
        }
        conv_outfile.close();
        std::cout << "Intermediate result for channel " << ch << " saved to " << conv_filename << std::endl;

        auto t_bn_start = std::chrono::high_resolution_clock::now();
        auto ct_bn = BatchNormOnCiphertext(
            cc, ct_conv,
            bn_gamma[ch], bn_beta[ch],
            bn_mean[ch], bn_var[ch],
            cc->GetEncodingParams()->GetBatchSize()
        );
        auto t_bn_end = std::chrono::high_resolution_clock::now();
        std::cout << "[Time] BatchNorm (channel " << ch << "): " << std::chrono::duration<double>(t_bn_end - t_bn_start).count() << " sec\n";

        Plaintext pt_bn_out;
        cc->Decrypt(keys.secretKey, ct_bn, &pt_bn_out);
        auto vec_bn = pt_bn_out->GetRealPackedValue();

        // 최종 결과(배치 정규화 이후) 저장
        std::string bn_filename = "conv1_output_channel_" + std::to_string(ch) + ".txt";
        std::ofstream bn_outfile(bn_filename);
        bn_outfile << std::fixed << std::setprecision(8);

        for (size_t i = 0; i < outH; i++) {
            for (size_t j = 0; j < outW; j++) {
                // 결과는 inputW 너비로 패딩되어 있으므로, 해당 인덱스에서 값을 가져옵니다.
                bn_outfile << vec_bn[i * inputW + j];
                if (j < outW - 1) {
                    bn_outfile << ",\n";
                }
            }
            if (i < outH - 1) {
                bn_outfile << ",\n";
            }
        }
        bn_outfile.close();
        std::cout << "Final result for channel " << ch << " saved to " << bn_filename << std::endl;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "[Time] Total Elapsed: " << std::chrono::duration<double>(t_end - t_start).count() << " sec\n";

    return 0;
}