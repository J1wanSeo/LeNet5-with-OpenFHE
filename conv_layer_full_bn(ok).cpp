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

Ciphertext<DCRTPoly> Conv2D_CKKS_SIMD(
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
    size_t outH = (inputH - filterH) / stride + 1; // height of output
    size_t outW = (inputW - filterW) / stride + 1; // width of output
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize(); // batchSize is same with number of slots

    std::vector<Ciphertext<DCRTPoly>> ct_outputs; // save the output of each filter

    for (size_t i = 0; i < outH; i++) { // for each output pixel
        for (size_t j = 0; j < outW; j++) { // for each output pixel
            size_t baseRow = i * stride; // base row of the filter
            size_t baseCol = j * stride; // base column of the filter
            size_t flatIdx = i * outW + j; // flat index of the output pixel

            std::vector<Ciphertext<DCRTPoly>> partials; // save the partial sum of each filter

            // fixed input with changing filter
            for (size_t dy = 0; dy < filterH; dy++) { // for each row of the filter
                for (size_t dx = 0; dx < filterW; dx++) { // for each column of the filter
                    size_t idx = dy * filterW + dx; // index of the filter
                    int rotAmount = (baseRow + dy) * inputW + (baseCol + dx); // rotation amount:location of the input pixel to multiply with the filter

                    Ciphertext<DCRTPoly> rotated = cc->EvalRotate(ct_input, rotAmount); // rotate the input
 
                    std::vector<double> wt(slotCount, 0.0); // weight vector initialized with 0
                    wt[0] = filter[idx]; // weight of the filter
                    auto pt_w = cc->MakeCKKSPackedPlaintext(wt); // plaintext of the weight
                    auto ct_mul = cc->EvalMult(rotated, pt_w); // multiply the rotated input with the weight

                    auto ct_shifted = cc->EvalRotate(ct_mul, -flatIdx);  // rotate the output to the correct position
                    partials.push_back(ct_shifted);
                }
            }

            std::vector<double> biasVec(slotCount, 0.0);
            biasVec[flatIdx] = bias;
            auto pt_bias = cc->MakeCKKSPackedPlaintext(biasVec);

            auto ct_sum = cc->EvalAddMany(partials);
            auto ct_with_bias = cc->EvalAdd(ct_sum, pt_bias);

            ct_outputs.push_back(ct_with_bias);
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
    auto filters = LoadFromTxt("../lenet_weights_epoch(10)/conv1_weight.txt");
    auto biases = LoadFromTxt("../lenet_weights_epoch(10)/conv1_bias.txt");
    auto bn_gamma = LoadFromTxt("../lenet_weights_epoch(10)/conv1_bn_gamma.txt");
    auto bn_beta  = LoadFromTxt("../lenet_weights_epoch(10)/conv1_bn_beta.txt");
    auto bn_mean  = LoadFromTxt("../lenet_weights_epoch(10)/conv1_bn_mean.txt");
    auto bn_var   = LoadFromTxt("../lenet_weights_epoch(10)/conv1_bn_var.txt");

    auto load_end = std::chrono::high_resolution_clock::now();
    //std::unordered_map<int, Ciphertext<DCRTPoly>> rotation_cache;
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
            size_t flatIdx = baseRow/stride * ((inputW-filterW)/stride + 1) + baseCol/stride;
            rot_index_set.insert(-static_cast<int>(flatIdx));
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

        start = std::chrono::high_resolution_clock::now();

        auto ct_conv = Conv2D_CKKS_SIMD(
            cc, ct_img, filter,
            inputH, inputW,
            filterH, filterW,
            stride,
            bias,
            keys.publicKey,
            rot_indices
        );


        end = std::chrono::high_resolution_clock::now();
        std::cout << "Elapsed time (sec): " << std::chrono::duration<double>(end - start).count() << std::endl;

        start = std::chrono::high_resolution_clock::now();

        auto ct_bn = BatchNormOnCiphertext(
            cc, ct_conv,
            bn_gamma[ch], bn_beta[ch],
            bn_mean[ch], bn_var[ch],
            outH * outW
        );

        end = std::chrono::high_resolution_clock::now();

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
