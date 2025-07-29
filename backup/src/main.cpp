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

    auto pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);


    vector<Ciphertext<DCRTPoly>> ct_input_channels = { ct_img };

    auto t0 = TimeNow();

    auto ct_conv1 = ConvBnLayer(cc, ct_input_channels, path, 32, 32, 5, 5, 1, 1, 6, 1, keys.publicKey, keys.secretKey);

    cout << "[Layer 1] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;

    SaveDecryptedConvOutput(cc, keys.secretKey, ct_conv1, 32, 32, "conv1_output");

    t0 = TimeNow();

    int mode = 2;
    auto ct_relu1 = ApplyApproxReLU4_All(cc, ct_conv1, mode);

    cout << "[Layer 1] ReLU elapsed: " << TimeNow() - t0 << " sec" << endl;

    SaveDecryptedConvOutput(cc, keys.secretKey, ct_relu1, 32, 32, "relu1_output");

    auto rot1 = GenerateRotationIndices(2, 2, 32);  // AvgPool용
    cc->EvalRotateKeyGen(keys.secretKey, rot1);

    t0 = TimeNow();
    auto ct_pool1 =  AvgPool2x2_MultiChannel_CKKS(cc, ct_relu1, 32, 32, rot1);
    cout << "[Layer 1] AvgPool elapsed: " << TimeNow() - t0 << " sec" << endl;

    SaveDecryptedConvOutput(cc, keys.secretKey, ct_pool1, 16, 16, "pool1_output");

    cout << "[LeNet-5 with OpenFHE] Forward Pass Completed and Output Saved." << endl;
    return 0;
}