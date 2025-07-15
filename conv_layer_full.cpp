#include "openfhe.h"
#include <vector>
#include <iostream>
#include <iomanip>

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
    size_t baseRow, size_t baseCol
) {

    std::vector<Ciphertext<DCRTPoly>> rotatedCts;

    for (size_t dy = 0; dy < filterH; dy++) {
        for (size_t dx = 0; dx < filterW; dx++) {
            size_t idx = dy * filterW + dx;
            int rotAmount = (baseRow + dy) * inputW + (baseCol + dx);
            auto rotated = cc->EvalRotate(ct_input, rotAmount);

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

    std::vector<double> outputs;

    for (size_t i = 0; i <= inputH - filterH; i += stride) {
        for (size_t j = 0; j <= inputW - filterW; j += stride) {
            auto ct_out = Conv2D_CKKS(cc, ct_input, filter, inputH, inputW,
                                      filterH, filterW, bias, pubkey, rot_indices,
                                      i, j);

            Plaintext pt_out;
            cc->Decrypt(secretKey, ct_out, &pt_out);
            pt_out->SetLength(1);
            outputs.push_back(pt_out->GetRealPackedValue()[0]);
        }
    }

    return outputs;
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

    // Load data from txt
    auto img = LoadFromTxt("../input_image.txt");
    auto filters = LoadFromTxt("../conv1_weight.txt");
    auto biases = LoadFromTxt("../conv1_bias.txt");

    Plaintext pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);

    size_t stride = 1;
    size_t inputH = 32, inputW = 32;
    size_t filterH = 5, filterW = 5;
    size_t channel = 6;
    size_t outH = (inputH - filterH) / stride + 1;
    size_t outW = (inputW - filterW) / stride + 1;

    std::set<int> rot_index_set;

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

    for (size_t ch = 0; ch < channel; ch++) {
        std::vector<double> filter(filters.begin() + ch * 25, filters.begin() + (ch + 1) * 25);
        double bias = biases[ch];

        auto outputs = Conv2D_FullOutput_Stride(
            cc, ct_img, filter,
            inputH, inputW,
            filterH, filterW,
            stride,
            bias,
            keys.publicKey,
            rot_indices,
            keys.secretKey
        );

        std::string filename = "conv1_output_channel_" + std::to_string(ch) + ".txt";
        std::ofstream outfile(filename);
        for (size_t i = 0; i < outH; i++) {
            for (size_t j = 0; j < outW; j++) {
                outfile << outputs[i * outW + j];
                if (j != outW - 1) outfile << ",";
            }
            outfile << "\n";
        }
        outfile.close();
    }

    return 0;
}
