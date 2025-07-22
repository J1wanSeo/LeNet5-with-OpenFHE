// main.cpp
#include "conv_bn_module.h"
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
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetRingDim(1 << 15);
    params.SetScalingModSize(35);
    params.SetBatchSize(4096);
    params.SetMultiplicativeDepth(15);
    // params.SetScalingTechnique(FIXEDMANUAL);  // ✅ 강력 추천


    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    string path = "../lenet_weights_epoch(10)";
    string input_file = "input_image.txt";

    auto img = LoadFromTxt(path + "/" + input_file);
    auto pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);

    for (size_t i = 0; i < 10; i++)
    std::cout << "img[" << i << "] = " << img[i] << std::endl;


    vector<Ciphertext<DCRTPoly>> ct_input_channels = { ct_img };

    auto t0 = TimeNow();
    auto ct_conv1 = ConvBnLayer(cc, ct_input_channels, path, 32, 32, 5, 5, 1, 1, 6, 1, keys.publicKey, keys.secretKey);
    cout << "[Layer 1] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;

    for (size_t ch = 0; ch < ct_conv1.size(); ++ch) {
        Plaintext pt;
        cc->Decrypt(keys.secretKey, ct_conv1[ch], &pt);
        pt->SetLength(28 * 28);  // 예시
        std::cout << "[Conv ch " << ch << "] Decode success, sample: " << pt->GetRealPackedValue()[0] << std::endl;
    }


    t0 = TimeNow();
    auto ct_relu1 = ApplyApproxReLU4_All(cc, ct_conv1);
    cout << "[Layer 1] ReLU elapsed: " << TimeNow() - t0 << " sec" << endl;

    auto rot1 = GenerateRotationIndices(2, 2, 32);
    cc->EvalRotateKeyGen(keys.secretKey, rot1);
    t0 = TimeNow();
    auto ct_pool1 = AvgPool2D_MultiChannel(cc, ct_relu1, 32, 32, keys.publicKey, rot1);
    cout << "[Layer 1] AvgPool elapsed: " << TimeNow() - t0 << " sec" << endl;

    // t0 = TimeNow();
    // auto ct_conv2 = ConvBnLayer(cc, ct_pool1, path, 16, 16, 5, 5, 1, 6, 16, 2, keys.publicKey, keys.secretKey);
    // cout << "[Layer 2] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;

    // t0 = TimeNow();
    // auto ct_relu2 = ApplyApproxReLU4_All(cc, ct_conv2);
    // cout << "[Layer 2] ReLU elapsed: " << TimeNow() - t0 << " sec" << endl;

    // auto rot2 = GenerateRotationIndices(2, 2, 16);
    // cc->EvalRotateKeyGen(keys.secretKey, rot2);
    // t0 = TimeNow();
    // auto ct_pool2 = AvgPool2D_MultiChannel(cc, ct_relu2, 16, 16, keys.publicKey, rot2);
    // cout << "[Layer 2] AvgPool elapsed: " << TimeNow() - t0 << " sec" << endl;

    size_t outH = 14, outW = 14;
    for (size_t ch = 0; ch < ct_pool1.size(); ++ch) {
        Plaintext pt;
        cc->Decrypt(keys.secretKey, ct_pool1[ch], &pt);
        pt->SetLength(outH * outW);
        auto vec = pt->GetRealPackedValue();

        ofstream out("openfhe_output_conv1_ch" + to_string(ch) + ".txt");
        for (size_t i = 0; i < outH; ++i) {
            for (size_t j = 0; j < outW; ++j) {
                out << fixed << setprecision(8) << vec[i * outW + j];
                if (j < outW - 1) out << ",\n";
            }
            out << "\n";
        }
    }

    cout << "[LeNet-5 with OpenFHE] Forward Pass Completed and Output Saved." << endl;
    return 0;
}