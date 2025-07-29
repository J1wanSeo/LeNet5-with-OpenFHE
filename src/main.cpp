// main.cpp
#include "conv_bn_module.h"
#include "relu.h"
// #include "fc_layer.h"
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
    auto pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);

    vector<Ciphertext<DCRTPoly>> ct_input_channels = {ct_img};

    // =======================
    // Layer 1: Conv1 + BN + ReLU + AvgPool
    // =======================
    auto t0 = TimeNow();
    auto ct_conv1 = ConvBnLayer(cc, ct_input_channels, path,
                                32, 32, 5, 5, 1,
                                1, 6, 1,
                                keys.publicKey, keys.secretKey);
    cout << "[Layer 1] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;
    SaveDecryptedConvOutput(cc, keys.secretKey, ct_conv1, 32, 32, "conv1_output");

    int mode = 2;
    auto ct_relu1 = ApplyApproxReLU4_All(cc, ct_conv1, mode);

    auto rot1 = GenerateRotationIndices(2, 2, 32);
    cc->EvalRotateKeyGen(keys.secretKey, rot1);
    auto ct_pool1 = AvgPool2x2_MultiChannel_CKKS(cc, ct_relu1, 32, 32, rot1);
    SaveDecryptedConvOutput(cc, keys.secretKey, ct_pool1, 16, 16, "pool1_output");

    // =======================
    // Layer 2: Conv2 + BN + ReLU + AvgPool
    // =======================
    t0 = TimeNow();
    auto ct_conv2 = ConvBnLayer(cc, ct_pool1, path,
                                16, 16, 5, 5, 1,
                                6, 16, 2,
                                keys.publicKey, keys.secretKey);
    cout << "[Layer 2] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;
    SaveDecryptedConvOutput(cc, keys.secretKey, ct_conv2, 16, 16, "conv2_output");

    auto ct_relu2 = ApplyApproxReLU4_All(cc, ct_conv2, mode);

    auto rot2 = GenerateRotationIndices(2, 2, 16);
    cc->EvalRotateKeyGen(keys.secretKey, rot2);
    auto ct_pool2 = AvgPool2x2_MultiChannel_CKKS(cc, ct_relu2, 16, 16, rot2);
    SaveDecryptedConvOutput(cc, keys.secretKey, ct_pool2, 8, 8, "pool2_output");

    // =======================
    // Layer 3: Conv3 + BN + ReLU (no Pool)
    // input: 16ch × 5×5 filter → output: 120 feature maps
    // =======================
    t0 = TimeNow();
    auto ct_conv3 = ConvBnLayer(cc, ct_pool2, path,
                                8, 8, 5, 5, 1,
                                16, 120, 3,
                                keys.publicKey, keys.secretKey);
    cout << "[Layer 3] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;
    SaveDecryptedConvOutput(cc, keys.secretKey, ct_conv3, 4, 4, "conv3_output");

    auto ct_relu3 = ApplyApproxReLU4_All(cc, ct_conv3, mode);

    // =======================
    // Layer 4: Fully Connected 120->84
    // =======================
    auto ct_fc1 = FCLayer_CKKS(cc, ct_relu3, path, 120, 84,
                               keys.publicKey, keys.secretKey, 1);
    SaveDecryptedConvOutput(cc, keys.secretKey, {ct_fc1}, 1, 84, "fc1_output");

    // =======================
    // Layer 5: FC 84->10 (Output)
    // =======================
    auto ct_fc2 = FCLayer_CKKS(cc, {ct_fc1}, path, 84, 10,
                               keys.publicKey, keys.secretKey, 2);
    SaveDecryptedConvOutput(cc, keys.secretKey, {ct_fc2}, 1, 10, "fc2_output");

    cout << "[LeNet-5 with OpenFHE] Forward Pass Completed and Output Saved." << endl;
    return 0;
}
