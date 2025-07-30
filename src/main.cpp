// main.cpp
#include "conv_bn_module.h"
#include "relu.h"
#include "fc_layer.h"
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
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);

    vector<Ciphertext<DCRTPoly>> ct_input_channels = {ct_img};

    std::set<int> all_rot_indices;

    auto t0 = TimeNow();
    // (1) Conv1: 5x5, input 32x32
    auto conv1_rot = GenerateRotationIndices(5, 5, 32);
    all_rot_indices.insert(conv1_rot.begin(), conv1_rot.end());
    auto conv2_rot = GenerateRotationIndices(5, 5, 28);
    all_rot_indices.insert(conv2_rot.begin(), conv2_rot.end());

    // (2) Conv1-Repack: 32x32 → 28x28
    auto repack1_rot = GenerateRepackRotationKeys(32, 32, 28, 28);
    all_rot_indices.insert(repack1_rot.begin(), repack1_rot.end());
    
    // (3) AvgPool1(SequentialPack): 28x28 input, 2x2, stride=2

    for (size_t dy = 0; dy < 2; dy++) {
        for (size_t dx = 0; dx < 2; dx++) {
            int rotAmount = dy * 28 + dx;
            all_rot_indices.insert(rotAmount);
        }
    }

    auto repack2_rot = GenerateRepackRotationKeys(56, 56, 28, 28);
    // auto repack2_rot2 = GenerateRepackRotationKeys(14, 28, 10, 10);
    all_rot_indices.insert(repack2_rot.begin(), repack2_rot.end());
    // all_rot_indices.insert(repack2_rot2.begin(), repack2_rot2.end());


    std::vector<int> rotIndices(all_rot_indices.begin(), all_rot_indices.end());
    cc->EvalRotateKeyGen(keys.secretKey, rotIndices);
    cout << "[Layer 1] Rotate KeyGen elapsed: " << TimeNow() - t0 << " sec" << endl;
    cout << "[INFO] All rotation key indices generated! (Total: " << rotIndices.size() << ")" << endl;
    


    // =======================
    // Layer 1: Conv1 + BN + ReLU + AvgPool
    // =======================
    t0 = TimeNow();
    auto ct_conv1 = ConvBnLayer(cc, ct_input_channels, path,
                                32, 32, 5, 5, 1,
                                1, 6, 1, 1,
                                keys.publicKey, keys.secretKey);
    cout << "[Layer 1] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_conv1, 32, 32, "conv1_output");

    int mode = 2;
    t0 = TimeNow();
    auto ct_relu1 = ApplyApproxReLU4_All(cc, ct_conv1, mode);
    cout << "[Layer 1] ReLU elapsed: " << TimeNow() - t0 << " sec" << endl;

    t0 = TimeNow(); 
    auto ct_repack1 = RepackConvolutionResult_MultiChannel(cc, ct_relu1, 32, 32, 28, 28);
    cout << "[Layer 1] Repack elapsed: " << TimeNow() - t0 << " sec" << endl;
    SaveDecryptedConvOutput(cc, keys.secretKey, ct_repack1, 28, 28, "repack1_output");

    t0 = TimeNow();
    auto ct_pool1 = AvgPool2x2_MultiChannel_CKKS(cc, ct_repack1, 28, 28);
    cout << "[Layer 1] AvgPool elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_pool1, 28, 28, "pool1_output");

    t0 = TimeNow();
    auto ct_repack1_2 = RepackConvolutionResult_MultiChannel(cc, ct_pool1, 56, 56, 28, 28);
    cout << "[Layer 1] Repack elapsed: " << TimeNow() - t0 << " sec" << endl;
    // // SaveDecryptedConvOutput(cc, keys.secretKey, ct_repack2, 28, 28, "repack2_output");


    // =======================
    // Layer 2: Conv2 + BN + ReLU + AvgPool
    // =======================
    t0 = TimeNow();
    auto ct_conv2 = ConvBnLayer(cc, ct_repack1, path,
                                14, 28, 5, 5, 1,
                                6, 16, 2, 2,
                                keys.publicKey, keys.secretKey);
    cout << "[Layer 2] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;
    SaveDecryptedConvOutput(cc, keys.secretKey, ct_conv2, 28, 28, "conv2_output");

    t0 = TimeNow();
    auto ct_relu2 = ApplyApproxReLU4_All(cc, ct_conv2, mode);
    cout << "[Layer 2] ReLU elapsed: " << TimeNow() - t0 << " sec" << endl;

    t0 = TimeNow(); 
    auto ct_repack2 = RepackConvolutionResult_MultiChannel(cc, ct_relu2, 28, 28, 10, 10);
    cout << "[Layer 2] Repack elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_repack2, 14, 14, "repack2_output");

    t0 = TimeNow();
    auto ct_pool2 = AvgPool2x2_MultiChannel_CKKS(cc, ct_relu2, 28, 28);
    cout << "[Layer 2] AvgPool elapsed: " << TimeNow() - t0 << " sec" << endl;
    SaveDecryptedConvOutput(cc, keys.secretKey, ct_pool2, 28, 28, "pool2_output");

    
    // t0 = TimeNow();
    // // 32x32 input에서 28x28 repack (좌상단)
    // // gather_indices 생성 (row-major 기준, stride/crop/ROI 모두 확장 가능)
    // gather_indices = GenerateRepackRotationKeys(56, 56, 20, 20); // INPUT: 64x64, OUTPUT: 28x28

    // // 필요한 rotation 인덱스 준비
    // cc->EvalRotateKeyGen(keys.secretKey, gather_indices);
    // cout << "[Layer 1] Rotate KeyGen elapsed: " << TimeNow() - t0 << " sec" << endl;

    // t0 = TimeNow();
    // auto ct_repack3 = RepackConvolutionResult_MultiChannel(cc, ct_pool2, 56, 56, 20, 20);
    // cout << "[Layer 1] Repack elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_repack3, 20, 20, "repack3_output");

    // // =======================
    // // Layer 3: Conv3 + BN + ReLU (no Pool)
    // // input: 16ch × 5×5 filter → output: 120 feature maps
    // // =======================
    // t0 = TimeNow();
    // auto ct_conv3 = ConvBnLayer(cc, ct_pool2, path,
    //                             8, 8, 5, 5, 1,
    //                             16, 120, 3,
    //                             keys.publicKey, keys.secretKey);
    // cout << "[Layer 3] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_conv3, 4, 4, "conv3_output");

    // auto ct_relu3 = ApplyApproxReLU4_All(cc, ct_conv3, mode);

    // // =======================
    // // Layer 4: Flatten 4x4x120 -> 1920
    // // =======================
    // auto ct_flatten = Flatten_CKKS(cc, ct_relu3, 4, 4, 120);
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_flatten, 1, 1920, "flatten_output");

    // // =======================
    // // Layer 4: Fully Connected 120->84
    // // =======================
    // auto ct_fc1 = GeneralFC_CKKS(cc, ct_relu3, path, 120, 84, 1, keys.publicKey);
    // SaveDecryptedConvOutput(cc, keys.secretKey, {ct_fc1}, 1, 84, "fc1_output");

    // // =======================
    // // Layer 5: FC 84->10 (Output)
    // // =======================
    // auto ct_fc2 = GeneralFC_CKKS(cc, {ct_fc1}, path, 84, 10, 2, keys.publicKey); // bn 없음 반영하기
    // SaveDecryptedConvOutput(cc, keys.secretKey, {ct_fc2}, 1, 10, "fc2_output");

    cout << "[LeNet-5 with OpenFHE] Forward Pass Completed and Output Saved." << endl;
    return 0;
}
