// main.cpp
#include "conv_bn_module.h"
#include "relu.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

using namespace std;

inline double TimeNow() {
    return chrono::duration<double>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main() {
    CCParams<CryptoContextCKKSRNS> params;
    // params.SetSecurityLevel(HEStd_NotSet);
    params.SetRingDim(1 << 16);
    params.SetScalingModSize(40);
    params.SetBatchSize(4096);
    params.SetMultiplicativeDepth(20);
    params.SetScalingTechnique(FLEXIBLEAUTO);


    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    string path = "../lenet_weights_epoch(10)";
    string input_file = "input_image.txt";

    auto img = LoadFromTxt(path + "/" + input_file);
    // std::cout << "[CHECK] First 5 values from input file: ";
    // for (int i = 0; i < 5; ++i) std::cout << img[i] << ", ";
    // std::cout << std::endl;

    // std::cout << "img.size():" << img.size() << std::endl;

    // std::cout << "=== 입력 데이터 슬롯 배치 확인 ===" << std::endl;
    // for (int i = 28; i < 35; i++) {
    //     std::cout << "Slot[" << i << "]: " << img[i] << std::endl;
    // }

    auto pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);


    vector<Ciphertext<DCRTPoly>> ct_input_channels = { ct_img };

    auto t0 = TimeNow();

    auto ct_conv1 = ConvBnLayer(cc, ct_input_channels, path, 32, 32, 5, 5, 1, 1, 6, 1, keys.publicKey, keys.secretKey);

    cout << "[Layer 1] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;


    // for (size_t ch = 0; ch < ct_conv1.size(); ++ch) {
    // std::cout << "[Conv1] ch " << ch << " - Level: " << ct_conv1[ch]->GetLevel()
    //           << ", Scale: " << ct_conv1[ch]->GetScalingFactor() << std::endl;
    // }

    
    // for (size_t ch = 0; ch < ct_conv1.size(); ch++) {
    //     Plaintext pt;
    //     cc->Decrypt(keys.secretKey, ct_conv1[ch], &pt);
    //     pt->SetLength(outH * outW);
    //     auto vec = pt->GetRealPackedValue();

    //     std::string filename = "conv1_output_channel_" + std::to_string(ch) + "_b4_relu.txt";
    //     std::ofstream out(filename);
    //     out << std::fixed << std::setprecision(8);

    //     for (size_t i = 0; i < outH; i++) {
    //         for (size_t j = 0; j < outW; j++) {
    //             out << vec[i * outW + j];
    //             if (j < outW - 1) out << ",\n";
    //         }
    //         out << ",\n";
    //     }
    //     std::cout << "Conv+BN result for channel " << ch << " saved to " << filename << std::endl;
    // }


    t0 = TimeNow();

    int mode = 2;
    auto ct_relu1 = ApplyApproxReLU4_All(cc, ct_conv1, mode);

    cout << "[Layer 1] ReLU elapsed: " << TimeNow() - t0 << " sec" << endl;

    for (size_t ch = 0; ch < ct_relu1.size(); ++ch) {
    std::cout << "[ReLU1] ch " << ch << " - Level: " << ct_relu1[ch]->GetLevel()
              << ", Scale: " << ct_relu1[ch]->GetScalingFactor() << std::endl;
    }

    // DEBUGGING AREA
    size_t inputH = 32, inputW = 32;  // Conv1 출력 크기 기준

    for (size_t ch = 0; ch < ct_relu1.size(); ++ch) {
        Plaintext pt;
        cc->Decrypt(keys.secretKey, ct_relu1[ch], &pt);
        // pt->SetLength(outH * outW);
        auto vec = pt->GetRealPackedValue();

        std::string filename = "conv1_output_channel_" + std::to_string(ch) + "_b4_avgpool.txt";
        std::ofstream out(filename);
        out << std::fixed << std::setprecision(8);

        for (size_t i = 0; i < inputH; i++) {
            for (size_t j = 0; j < inputW; j++) {
                out << vec[i * inputW + j];
                if (j < inputW - 1) out << ",\n";
            }
            out << "\n";
        }
        std::cout << "Relu1 Results are " << ch << " saved to " << filename << std::endl;
    }

    auto rot1 = GenerateRotationIndices(2, 2, 32);  // AvgPool용
    cc->EvalRotateKeyGen(keys.secretKey, rot1);

    t0 = TimeNow();
    auto ct_pool1 = AvgPool2D_MultiChannel(cc, ct_relu1, 32, 32, keys.publicKey, rot1);
    cout << "[Layer 1] AvgPool elapsed: " << TimeNow() - t0 << " sec" << endl;

    size_t outH_pool = 32, outW_pool = 32;
    for (size_t ch = 0; ch < ct_pool1.size(); ++ch) {
        Plaintext pt;
        cc->Decrypt(keys.secretKey, ct_pool1[ch], &pt);
        pt->SetLength(outH_pool * outW_pool);
        auto vec = pt->GetRealPackedValue();

        ofstream out("openfhe_output_conv1_ch" + to_string(ch) + ".txt");
        for (size_t i = 0; i < outH_pool; ++i) {
            for (size_t j = 0; j < outW_pool; ++j) {
                out << fixed << setprecision(8) << vec[i * outW_pool + j];
                if (j < outW_pool - 1) out << ",\n";
            }
            out << "\n";
        }
    }

    cout << "[LeNet-5 with OpenFHE] Forward Pass Completed and Output Saved." << endl;
    return 0;
}