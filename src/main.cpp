// main.cpp
#include "conv_bn_module.h"
#include "relu.h"
#include "fc_layer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

using namespace std;


int main() {
    int relu_mode;

    std::cout << " =============================================\n";
    std::cout << "|           Select ReLU Mode                  |\n";
    std::cout << " =============================================\n";
    std::cout << std::left;
    std::cout << " 0 : x (linear function)\n";
    std::cout << " 1 : x^2 (Square function)\n";
    std::cout << " 2 : CryptoNet (0.25 + 0.5x + 0.125x^2)\n";
    std::cout << " 3 : quad (4th degree polynomial approx.)\n";
    std::cout << " 4 : student polynomial (custom)\n";
    std::cout << "---------------------------------------------\n";
    std::cout << "Enter your choice (0 - 4): ";

    std::cin >> relu_mode;
    int total_depth = CalculateMultiplicativeDepth(relu_mode);
    std::cout << "You selected mode: " << relu_mode << std::endl;
    

    CCParams<CryptoContextCKKSRNS> params;
    // params.SetRingDim(1 << 15);
    // params.SetScalingModSize(40);
    params.SetBatchSize(1 << 10);
    params.SetMultiplicativeDepth(total_depth);
    params.SetScalingTechnique(FLEXIBLEAUTO);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    // cc->Disable(BOOTSTRAPPED);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    string path = "../pytorch_LeNet5/parameters_standard";
    string input_file = "input_image.txt";

    auto img = LoadFromTxt("../" + input_file);
    auto pt_img = cc->MakeCKKSPackedPlaintext(img);
    auto ct_img = cc->Encrypt(keys.publicKey, pt_img);

    vector<Ciphertext<DCRTPoly>> ct_input_channels = {ct_img};

    std::set<int> all_rot_indices;

    // logN (log2 of ring dimension)
    size_t ringDim = cc->GetRingDimension();  // 2^logN
    size_t logN = static_cast<size_t>(std::log2(ringDim));

    // logQ (log2 scale)
    // double scale = cc->GetCryptoParameters()->GetScalingModSize();    // scale = 2^logQ
    // double logQ = std::log2(scale);

    // SlotSize
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    std::cout << "\033[2J\033[H";
    std::cout << "Estimated total multiplicative depth: " << total_depth << std::endl;
    std::cout << "[OpenFHE Info] logN (ring dimension exponent): " << logN << std::endl;
    // std::cout << "[OpenFHE Info] logQ (scale exponent): " << logQ << std::endl;
    std::cout << "[OpenFHE Info] SlotSize (batch size): " << slotCount << std::endl;

    

    auto t0 = TimeNow();

    // (1) Conv1: 5x5, input 32x32, Conv2: 5x5, input 14x14
    auto conv1_rot = GenerateRotationIndices(5, 5, 32, 1);
    all_rot_indices.insert(conv1_rot.begin(), conv1_rot.end());
    auto conv2_rot = GenerateRotationIndices(5, 5, 32, 2);
    all_rot_indices.insert(conv2_rot.begin(), conv2_rot.end());

    // (2) Conv1-Repack: 32x32 → 28x28
    // auto ReAlign1_rot = GenerateReAlignRotationKeys(32, 32, 28, 28, 1);
    // all_rot_indices.insert(ReAlign1_rot.begin(), ReAlign1_rot.end());
    // auto ReAlign2_rot = GenerateReAlignRotationKeys(28, 28, 20, 20, 1);
    // all_rot_indices.insert(ReAlign2_rot.begin(), ReAlign2_rot.end());


    // (3) AvgPool1(SequentialPack): 28x28 input, 2x2, stride=2

    // auto pool1_rot = GenerateRotationIndices(2, 2, 28, 1);
    // all_rot_indices.insert(pool1_rot.begin(), pool1_rot.end());
    // auto pool2_rot = GenerateRotationIndices(2, 2, 20, 2);
    // all_rot_indices.insert(pool2_rot.begin(), pool2_rot.end());


    // 30 keys
    auto validIndices = GetValidSlotIndices(20, 32, 4);
    auto flattenRot = GetFlattenRotationIndices(validIndices);
    // auto concatRot = GetConcatRotationIndices(16, validIndices.size());
    all_rot_indices.insert(flattenRot.begin(), flattenRot.end());
    // all_rot_indices.insert(concatRot.begin(), concatRot.end());


    for (size_t k = 1; k < 1024; k <<= 1) {
        all_rot_indices.insert(k);        
    }


    std::vector<int> rotIndices(all_rot_indices.begin(), all_rot_indices.end());
    cc->EvalRotateKeyGen(keys.secretKey, rotIndices);
    cout << "[Layer 1] Rotate KeyGen elapsed: " << TimeNow() - t0 << " sec" << endl;
    cout << "[INFO] All rotation key indices generated! (Total: " << rotIndices.size() << ")" << endl;
    


    // =======================
    // Layer 1: Conv1 + BN + ReLU + AvgPool 3 + 1 + 1
    // =======================
    t0 = TimeNow(); //3
    auto ct_conv1 = ConvBnLayer(cc, ct_input_channels, path,
                                32, 32, 5, 5, 1,
                                1, 6, 1, 1,
                                keys.publicKey, keys.secretKey);
    cout << "[Layer 1] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_conv1, 32, 32, "conv1_output");

    t0 = TimeNow(); // 1
    auto ct_relu1 = ApplyApproxReLU4_All(cc, ct_conv1, relu_mode);
    cout << "[Layer 1] ReLU elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_relu1, 32, 32, "relu1_output");

    // t0 = TimeNow();  // 1
    // auto ct_repack1 = ReAlignConvolutionResult_MultiChannel(cc, ct_relu1, 32, 32, 28, 28, 1);
    // cout << "[Layer 1] Realignment elapsed: " << TimeNow() - t0 << " sec" << endl;
    // // SaveDecryptedConvOutput(cc, keys.secretKey, ct_repack1, 28, 28, "realign1_output");

    t0 = TimeNow(); // 1
    auto ct_pool1 = AvgPool2x2_MultiChannel_CKKS(cc, ct_relu1, 28, 32, 1);
    cout << "[Layer 1] AvgPool elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_pool1, 28, 32, "pool1_output");

    // =======================
    // Layer 2: Conv2 + BN + ReLU + AvgPool
    // =======================
    t0 = TimeNow(); // 3
    auto ct_conv2 = ConvBnLayer(cc, ct_pool1, path,
                                28, 32, 5, 5, 1,
                                6, 16, 2, 2,
                                keys.publicKey, keys.secretKey);
    cout << "[Layer 2] Conv+BN elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_conv2, 28, 28, "conv2_output");

    t0 = TimeNow(); // 1
    auto ct_relu2 = ApplyApproxReLU4_All(cc, ct_conv2, relu_mode);
    cout << "[Layer 2] ReLU elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_relu2, 28, 28, "relu2_output");

    // t0 = TimeNow();  // 1
    // auto ct_repack2 = ReAlignConvolutionResult_MultiChannel(cc, ct_relu2, 28, 28, 20, 20, 1);
    // cout << "[Layer 2] Repack elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_repack2, 20, 20, "repack2_output");

    t0 = TimeNow();//1
    auto ct_pool2 = AvgPool2x2_MultiChannel_CKKS(cc, ct_relu2, 20, 32, 2);
    cout << "[Layer 2] AvgPool elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_pool2, 20, 20, "pool2_output");

    t0 = TimeNow(); // 1
    Ciphertext<DCRTPoly> ct_flat = FlattenCiphertexts(cc, ct_pool2, keys.secretKey);
    cout << "[Flatten Layer] Flatten elapsed: " << TimeNow() - t0 << " sec" << endl;

    // 단일 ciphertext를 벡터에 담아서 전달
    std::vector<Ciphertext<DCRTPoly>> ct_flattened = { ct_flat };

    // SaveDecryptedConvOutput(cc, keys.secretKey, ct_flattened, 1, 20 * 20, "flatten_output");

    t0 = TimeNow(); // 2
    auto ct_fc1 = GeneralFC_CKKS(cc, ct_flat, path, 400, 120, 1, keys.publicKey);
    cout << "[Layer 3] FC + BN elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedFCOutput(cc, keys.secretKey, ct_fc1, 120, "fc1_output");

    t0 = TimeNow(); // 1
    auto ct_relu3 = ApplyApproxReLU4_All(cc, {ct_fc1}, relu_mode);
    cout << "[Layer 3] ReLU elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedFCOutput(cc, keys.secretKey, ct_relu3[0], 120, "relu3_output");
    

    // =======================
    // Layer 5: FC 84->10 (Output)
    // =======================
    t0 = TimeNow(); // 2
    auto ct_fc2 = GeneralFC_CKKS(cc, ct_relu3[0], path, 120, 84, 2, keys.publicKey);
    cout << "[Layer 4] FC + BN elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedFCOutput(cc, keys.secretKey, ct_fc2, 84, "fc2_output");
    
    t0 = TimeNow(); // 1
    auto ct_relu4 = ApplyApproxReLU4_All(cc, {ct_fc2}, relu_mode);
    cout << "[Layer 4] ReLU elapsed: " << TimeNow() - t0 << " sec" << endl;
    // SaveDecryptedFCOutput(cc, keys.secretKey, ct_relu4[0], 84, "relu4_output");

    // =======================
    // Layer 5: FC 84->10 (Output)
    // =======================
    // 2
    t0 = TimeNow();
    auto ct_fc3 = GeneralFC_wo_BN_CKKS(cc, ct_relu4[0], path, 84, 10, 3, keys.publicKey); // bn 없음 반영하기
    cout << "[Layer 5] FC elapsed: " << TimeNow() - t0 << " sec" << endl;
    SaveDecryptedFCOutput(cc, keys.secretKey, ct_fc3, 10, "fc3_output");

    cout << "[LeNet-5 with OpenFHE] Forward Pass Completed and Output Saved." << endl;
    return 0;   
}
