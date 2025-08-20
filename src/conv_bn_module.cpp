// conv_bn_module.cpp
#include "conv_bn_module.h"
#include <chrono>
#include <mutex>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <iostream>
#include <thread>
#include <future>

std::vector<double> LoadFromTxt(const std::string& filename) {
    std::ifstream infile(filename);
    std::vector<double> data;
    std::string content;
    std::getline(infile, content);
    std::stringstream ss(content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) data.push_back(std::stod(token));
    }
    return data;
}


std::vector<int> GenerateRotationIndices(size_t filterH, size_t filterW, size_t inputW, size_t interleave) {
    std::set<int> rotSet;
    for (size_t dy = 0; dy < filterH; dy++) {
        for (size_t dx = 0; dx < filterW; dx++) {
        int rotAmount = dy * inputW * interleave + dx * interleave;
        rotSet.insert(rotAmount);
        // rotSet.insert(-rotAmount);
        }
    }
    return std::vector<int>(rotSet.begin(), rotSet.end());
}

std::vector<int> GetFlattenRotationIndices(const std::vector<int>& validIndices) {
    std::vector<int> rotKeys;
    for (size_t i = 1; i < validIndices.size(); i++) {
        rotKeys.push_back((validIndices[i]-i));
    }
    return rotKeys;
}

std::vector<int> GetConcatRotationIndices(size_t numCts, size_t perCtValidCount) {
    std::vector<int> rotKeys;
    for (size_t i = 1; i < numCts; ++i) {
        rotKeys.push_back(-static_cast<int>(perCtValidCount * i));
    }
    return rotKeys;
}

Ciphertext<DCRTPoly> GeneralBatchNorm_CKKS(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_input,
    double gamma, double beta,
    double mean, double var,
    size_t slotCount) {

    double eps = 1e-5;
    double a = gamma / std::sqrt(var + eps);
    double b = beta - a * mean;

    auto pt_a = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, a));
    auto pt_b = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, b));
    
    auto scaled_mul = cc->EvalMult(ct_input, pt_a);
    scaled_mul = cc->Rescale(scaled_mul);
    return cc->EvalAdd(scaled_mul, pt_b);

}

std::vector<Ciphertext<DCRTPoly>> ConvBnLayer(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_input_channels,
    const std::string& pathPrefix,
    size_t inputH, size_t inputW,
    size_t filterH, size_t filterW,
    size_t stride,
    size_t in_channels, size_t out_channels,
    size_t layerIndex,
    size_t interleave,
    const PublicKey<DCRTPoly>& publicKey,
    const PrivateKey<DCRTPoly>& secretKey) {

    std::string layerPrefix = "conv" + std::to_string(layerIndex);
    auto filterPath = pathPrefix + "/" + layerPrefix + "_weight.txt";
    auto biasPath   = pathPrefix + "/" + layerPrefix + "_bias.txt";
    auto gammaPath  = pathPrefix + "/" + layerPrefix + "_bn_gamma.txt";
    auto betaPath   = pathPrefix + "/" + layerPrefix + "_bn_beta.txt";
    auto meanPath   = pathPrefix + "/" + layerPrefix + "_bn_mean.txt";
    auto varPath    = pathPrefix + "/" + layerPrefix + "_bn_var.txt";

    auto filters = LoadFromTxt(filterPath);
    auto biases  = LoadFromTxt(biasPath);
    auto gammas  = LoadFromTxt(gammaPath);
    auto betas   = LoadFromTxt(betaPath);
    auto means   = LoadFromTxt(meanPath);
    auto vars    = LoadFromTxt(varPath);

    // size_t outH = (inputH - filterH) / stride + 1;
    // size_t outW = (inputW - filterW) / stride + 1;

    std::vector<Ciphertext<DCRTPoly>> outputs(out_channels);

    std::mutex cout_mutex;

    #pragma omp parallel for schedule(dynamic)
    for (size_t out_ch = 0; out_ch < out_channels; out_ch++) {
        Ciphertext<DCRTPoly> ct_sum; // lazy-init
        double t_start = TimeNow();
        for (size_t in_ch = 0; in_ch < in_channels; in_ch++) {
            size_t base = (out_ch * in_channels + in_ch) * filterH * filterW;
            const std::vector<double> filter(
                filters.begin() + base,
                filters.begin() + base + filterH * filterW
            );
            // ===== 개선 핵심: 스파스 필터 skip하여 rotation 줄이기 =====
            bool first_partial = true;
            Ciphertext<DCRTPoly> ct_filtered_sum; // 각 input-ch에 대한 partial sum
            for (size_t i = 0; i < filter.size(); i++) {
                double w = filter[i];
                if (std::abs(w) < 1e-8) continue; // 0 weight skip
                size_t dy = i / filterW;
                size_t dx = i % filterW;
                int rotAmount = static_cast<int>(dy * inputW * interleave + dx * interleave);
                auto rotated = cc->EvalRotate(ct_input_channels[in_ch], rotAmount);
                // Masking & weight를 동시에 곱해서 바로 partial에 누적
                auto masked = cc->EvalMult(rotated, w); // w: double -> Plaintext 자동 처리
                if (first_partial) {
                    ct_filtered_sum = masked;
                    first_partial = false;
                } else {
                    ct_filtered_sum = cc->EvalAdd(ct_filtered_sum, masked);
                }
            }
        if (!first_partial) {
            // bias, bn은 기존대로 진행
            if (in_ch == 0) {
                ct_sum = ct_filtered_sum;
            } else {
                ct_sum = cc->EvalAdd(ct_sum, ct_filtered_sum);
            }
        }
    }

        double t_conv_end = TimeNow();

        auto bias = biases[out_ch];
        std::vector<double> bias_vec(cc->GetEncodingParams()->GetBatchSize(), bias);
        auto pt_bias = cc->MakeCKKSPackedPlaintext(bias_vec);
        ct_sum = cc->EvalAdd(ct_sum, pt_bias);

        double t_bias_end = TimeNow();

        auto ct_bn = GeneralBatchNorm_CKKS(cc, ct_sum, gammas[out_ch], betas[out_ch], means[out_ch], vars[out_ch], cc->GetEncodingParams()->GetBatchSize());
        outputs[out_ch] = ct_bn;

        double t_bn_end = TimeNow();

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[ConvBnLayer] OutCh " << out_ch
                      << " Conv elapsed: " << (t_conv_end - t_start) << " sec, "
                      << "Bias elapsed: " << (t_bias_end - t_conv_end) << " sec, "
                      << "BatchNorm elapsed: " << (t_bn_end - t_bias_end) << " sec"
                      << std::endl;
        }
    }

    return outputs;
}

Ciphertext<DCRTPoly> RotateByIndex(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct,
    int rot,              // 음수 혹은 양수 회전 인덱스
    int slotCount
) {
    int posRot = (rot >= 0) ? rot : (slotCount + rot);

    Ciphertext<DCRTPoly> result = ct;

    for (int k = 1 << 30; k > 0; k >>= 1) {
        if ((posRot & k) != 0) {
            result = cc->EvalRotate(result, k);
        }
    }
    return result;
}


// 1) 유효 슬롯 인덱스 출력 포함
std::vector<int> GetValidSlotIndices(size_t rows, size_t cols, size_t colStride) {
    std::vector<int> validIndices;
    for (size_t r = 0; r < rows; r+= colStride) {
        for (size_t c = 0; c < rows; c += colStride) {
            int idx = static_cast<int>(r * cols + c);
            validIndices.push_back(idx);
        }
    }

    std::cout << "[DEBUG] GetValidSlotIndices: total valid slots = " << validIndices.size() << std::endl;
    // 첫 10개만 출력 (필요하면 늘리거나 줄이기)
    std::cout << "[DEBUG] validIndices sample: ";
    for (size_t i = 0; i < std::min<size_t>(50, validIndices.size()); ++i) {
        std::cout << validIndices[i] << " ";
    }
    std::cout << std::endl;

    return validIndices;
}

// 2) CompressValidSlots에 진입과 주요 작업 로그 추가
Ciphertext<DCRTPoly> CompressValidSlots(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct,
    const std::vector<int>& validIndices) {

    std::cout << "[DEBUG] CompressValidSlots: start, validIndices size = " << validIndices.size() << std::endl;

    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    Ciphertext<DCRTPoly> compressed;

    for (size_t i = 0; i < validIndices.size(); ++i) {
        int src = validIndices[i];
        if (src < 0 || src >= (int)slotCount) {
            std::cout << "[WARN] CompressValidSlots: valid index out of range: " << src << std::endl;
            continue;
        }

        // 1. 슬롯 하나만 살리는 마스크
        std::vector<double> mask(slotCount, 0.0);
        mask[src] = 1.0;
        auto pt_one = cc->MakeCKKSPackedPlaintext(mask);

        // 2. 해당 슬롯만 남긴 후
        auto ct_single = cc->EvalMult(ct, pt_one);

        // 3. i번 슬롯으로 이동시키기
        int rotAmount = src - static_cast<int>(i); // 좌측으로 rotAmount만큼 회전
        auto ct_rotated = cc->EvalRotate(ct_single, rotAmount);
        // auto ct_rotated = RotateByIndex(cc, ct_single, rotAmount, cc->GetEncodingParams()->GetBatchSize());

        // 4. 누적 합
        if (i == 0)
            compressed = ct_rotated;
        else
            compressed = cc->EvalAdd(compressed, ct_rotated);
    }

    // std::cout << "[DEBUG] CompressValidSlots: completed compression" << std::endl;
    return compressed;
}


// 3) ConcatenateCiphertexts에 로그 추가
Ciphertext<DCRTPoly> ConcatenateCiphertexts(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& cts,
    size_t perCtValidCount) {

    Ciphertext<DCRTPoly> result = cts[0];

    for (size_t i = 1; i < cts.size(); ++i) {
        int rot = -static_cast<int>(perCtValidCount * i); // rotate to right
        
        // auto shifted = cc->EvalRotate(cts[i], rot);
        auto shifted = RotateByIndex(cc, cts[i], rot, cc->GetEncodingParams()->GetBatchSize());
        result = cc->EvalAdd(result, shifted);
    }
    return result;
}

// 4) Flatten 전체 프로세스에 단계별 로그 추가
Ciphertext<DCRTPoly> FlattenCiphertexts(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_vec,
    const PrivateKey<DCRTPoly>& secretKey // remove after debugging
) {

    std::cout << "[DEBUG] FlattenCiphertexts: start, number of input ciphertexts = " << ct_vec.size() << std::endl;

    std::vector<int> validIndices = GetValidSlotIndices(20, 32, 4);

    std::vector<Ciphertext<DCRTPoly>> compressed_cts(ct_vec.size());
    #pragma omp parallel for
    for (size_t i = 0; i < (int)ct_vec.size(); i++) {
        // std::cout << "[DEBUG] FlattenCiphertexts: compressing ciphertext " << i << std::endl;
        compressed_cts[i] = CompressValidSlots(cc, ct_vec[i], validIndices);
        // if(i == 1){
        // std::vector<Ciphertext<DCRTPoly>> ct_test = {compressed_cts[i] };
        // SaveDecryptedConvOutput(cc, secretKey, ct_test, 1, 20 * 20, "compressed");
        // }
    }

    size_t perCtValidCount = validIndices.size();
    std::cout << "[DEBUG] FlattenCiphertexts: per ciphertext valid slot count = " << perCtValidCount << std::endl;

    auto flattened = ConcatenateCiphertexts(cc, compressed_cts, perCtValidCount);

    // std::cout << "[DEBUG] FlattenCiphertexts: completed flattening" << std::endl;
    return flattened;
}


std::vector<Ciphertext<DCRTPoly>> AvgPool2x2_MultiChannel_CKKS(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    size_t inputH, size_t inputW, size_t interleave) {

    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    size_t filterH = 2, filterW = 2, stride = 2;
    size_t outH = (inputH - filterH) / (stride * interleave) + 1;
    size_t outW = (inputH - filterW) / (stride * interleave) + 1;

    // stride 2 고려한 mask 생성 (모든 채널 공통 사용 가능)
    std::vector<double> mask(slotCount, 0.0);
    for (size_t i = 0; i < outH; i++) {
        for (size_t j = 0; j < outW; j++) {
            size_t output_slot = (i * stride * interleave) * inputW + (j * stride * interleave);
            if (output_slot < slotCount) {
                mask[output_slot] = 0.25; // mask[output_slot] = 0.25;
            }
        }
    }
    auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);

    // 2x2 평균 filter weight = 1/4
    // double weight = 0.25;

    // 채널별 AvgPool 수행
    std::vector<Ciphertext<DCRTPoly>> pooled(ct_channels.size());

    #pragma omp parallel for
    for (size_t i = 0; i < ct_channels.size(); i++) {
        const auto& ct_input = ct_channels[i];
        std::vector<Ciphertext<DCRTPoly>> partials;

        for (size_t dy = 0; dy < filterH; dy++) {
            for (size_t dx = 0; dx < filterW; dx++) {
                int rotAmount = dy * inputW * interleave + dx * interleave;
                auto rotated = cc->EvalRotate(ct_input, rotAmount);

                // auto masked_rotated = cc->EvalMult(rotated, pt_mask);
                // masked_rotated = cc->Rescale(masked_rotated);

                auto ct_weighted = cc->EvalMult(rotated, pt_mask);
                // ct_weighted = cc->Rescale(ct_weighted);

                partials.push_back(ct_weighted);
            }
        }

        pooled[i] = cc->EvalAddMany(partials);
    }

    return pooled;
}


void SaveDecryptedConvOutput(
    CryptoContext<DCRTPoly> cc,
    const PrivateKey<DCRTPoly>& sk,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    size_t outH, size_t outW,
    const std::string& prefix) {

    for (size_t ch = 0; ch < ct_channels.size(); ++ch) {
        // 복호화
        Plaintext pt;
        cc->Decrypt(sk, ct_channels[ch], &pt);
        pt->SetLength(outH * outW);
        auto vec = pt->GetRealPackedValue();

        // 파일명 구성
        std::string filename = prefix + "_channel_" + std::to_string(ch) + ".txt";
        std::ofstream out(filename);
        out << std::fixed << std::setprecision(8);

        // 데이터 저장
        for (size_t i = 0; i < outH; i++) {
            for (size_t j = 0; j < outW; j++) {
                out << vec[i * outW + j];
                if (j < outW - 1) out << ",\n";
            }
            out << ",\n";
        }

        std::cout << "[INFO] Output saved: " << filename << std::endl;
    }
}

int CalculateMultiplicativeDepth(int relu_mode) {
    // 기본값: relu multiplicative depth
    int relu_depth = 4; 
    
    switch (relu_mode) {
        case 0: relu_depth = 0; break;  // linear x, 사실 곱셈 없음
        case 1: relu_depth = 2; break;  // x^2
        case 2: relu_depth = 2; break;  // CryptoNet
        case 3: relu_depth = 4; break;  // quad
        case 4: relu_depth = 4; break;  // student polynomial
        default: relu_depth = 4; break;
    }

    int depth = 0;

    // # of mult depth/layer * # of layers 
    depth += 2 * 2;   // conv + bn
    depth += 1 * 2;   // avgpool
    depth += relu_depth * 4;  // relu (relu_mode에 따라 다름)
    depth += 3 * 2;   // fc + bn
    depth += 2 * 1;   // fc
    depth += 1 * 1;   // flatten
    // depth += 1;
    return depth;
}
