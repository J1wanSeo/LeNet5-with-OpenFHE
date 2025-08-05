// main_unified.cpp
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
    params.SetRingDim(1 << 16);
    params.SetScalingModSize(40);
    params.SetBatchSize(1 << 15);
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
    auto pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_input = cc->Encrypt(keys.publicKey, pt_img);

    int relu_mode;
    cout << "Choose ReLU mode (0=square, 1=CryptoNet, 2=quad): ";
    cin >> relu_mode;

    double t0, elapsed;

    t0 = TimeNow();
    auto rot1 = GenerateUnifiedConvRotationIndices(5, 5, 32, 1);
    cc->EvalRotateKeyGen(keys.secretKey, rot1);
    elapsed = TimeNow() - t0;
    cout << "[RotateKeyGen Conv1] " << elapsed << " sec" << endl;

    t0 = TimeNow();
auto pool_rot1 = GenerateUnifiedConvRotationIndices(2, 2, 28, 6);
cc->EvalRotateKeyGen(keys.secretKey, pool_rot1);
elapsed = TimeNow() - t0;
cout << "[RotateKeyGen Pool1] " << elapsed << " sec" << endl;

// Layer 1
    t0 = TimeNow();
    auto ct_conv1 = UnifiedConvBnLayer(cc, ct_input, path, 32, 32, 5, 5, 1, 1, 6, 1, keys.publicKey, keys.secretKey);
    SaveDecryptedPackedOutput(cc, keys.secretKey, ct_conv1, 28, 28, 6, "conv1_output");
    elapsed = TimeNow() - t0;
    cout << "[Layer 1] Conv+BN: " << elapsed << " sec" << endl;

    t0 = TimeNow();
    auto ct_relu1 = UnifiedReLU(cc, ct_conv1, relu_mode);
    elapsed = TimeNow() - t0;
    cout << "[Layer 1] ReLU: " << elapsed << " sec" << endl;

    t0 = TimeNow();
    auto ct_pool1 = UnifiedAvgPool2x2(cc, ct_relu1, 28, 28, 6);
    elapsed = TimeNow() - t0;
    cout << "[Layer 1] AvgPool: " << elapsed << " sec" << endl;

    t0 = TimeNow();
auto pool_rot2 = GenerateUnifiedConvRotationIndices(2, 2, 10, 16);
cc->EvalRotateKeyGen(keys.secretKey, pool_rot2);
elapsed = TimeNow() - t0;
cout << "[RotateKeyGen Pool2] " << elapsed << " sec" << endl;

// Layer 2
    t0 = TimeNow();
    auto rot2 = GenerateUnifiedConvRotationIndices(5, 5, 14, 6);
    cc->EvalRotateKeyGen(keys.secretKey, rot2);
    elapsed = TimeNow() - t0;
    cout << "[RotateKeyGen Conv2] " << elapsed << " sec" << endl;

    t0 = TimeNow();
    auto ct_conv2 = UnifiedConvBnLayer(cc, ct_pool1, path, 14, 14, 5, 5, 1, 6, 16, 2, keys.publicKey, keys.secretKey);
    SaveDecryptedPackedOutput(cc, keys.secretKey, ct_conv2, 10, 10, 16, "conv2_output");
    elapsed = TimeNow() - t0;
    cout << "[Layer 2] Conv+BN: " << elapsed << " sec" << endl;

    t0 = TimeNow();
    auto ct_relu2 = UnifiedReLU(cc, ct_conv2, relu_mode);
    elapsed = TimeNow() - t0;
    cout << "[Layer 2] ReLU: " << elapsed << " sec" << endl;

    t0 = TimeNow();
    auto ct_pool2 = UnifiedAvgPool2x2(cc, ct_relu2, 10, 10, 16);
    elapsed = TimeNow() - t0;
    cout << "[Layer 2] AvgPool: " << elapsed << " sec" << endl;

    // Layer 3
    t0 = TimeNow();
    auto rot3 = GenerateUnifiedConvRotationIndices(5, 5, 5, 16);
    cc->EvalRotateKeyGen(keys.secretKey, rot3);
    elapsed = TimeNow() - t0;
    cout << "[RotateKeyGen Conv3] " << elapsed << " sec" << endl;

    t0 = TimeNow();
    auto ct_conv3 = UnifiedConvBnLayer(cc, ct_pool2, path, 5, 5, 5, 5, 1, 16, 120, 3, keys.publicKey, keys.secretKey);
    SaveDecryptedPackedOutput(cc, keys.secretKey, ct_conv3, 1, 1, 120, "conv3_output");
    elapsed = TimeNow() - t0;
    cout << "[Layer 3] Conv+BN: " << elapsed << " sec" << endl;

    t0 = TimeNow();
    auto ct_relu3 = UnifiedReLU(cc, ct_conv3, relu_mode);
    elapsed = TimeNow() - t0;
    cout << "[Layer 3] ReLU: " << elapsed << " sec" << endl;

    t0 = TimeNow();
    SaveDecryptedPackedOutput(cc, keys.secretKey, ct_relu3, 1, 1, 120, "final_output");
    // save intermediate results too
    SaveDecryptedPackedOutput(cc, keys.secretKey, ct_pool1, 14, 14, 6, "pool1_output");
    SaveDecryptedPackedOutput(cc, keys.secretKey, ct_pool2, 5, 5, 16, "pool2_output");
    elapsed = TimeNow() - t0;
    cout << "[Save Output] " << elapsed << " sec" << endl;

    cout << "[OpenFHE LeNet] Completed." << endl;
    return 0;
}
