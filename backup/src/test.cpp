    #include "openfhe.h"
    #include "conv_bn_module.h"
    #include <iostream>
    #include <vector>
    #include <iomanip>

    using namespace lbcrypto;


    std::vector<Ciphertext<DCRTPoly>> AvgPool2D_MultiChannel_3x3(
        CryptoContext<DCRTPoly> cc,
        const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
        size_t inputH, size_t inputW,
        const PublicKey<DCRTPoly>& pk,
        const std::vector<int>& rotIndices) {
    
        std::vector<Ciphertext<DCRTPoly>> pooled;
        size_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    
        size_t outH = inputH / 2;
        size_t outW = inputW / 2;
    
        for (const auto& ct : ct_channels) {
            std::vector<Ciphertext<DCRTPoly>> outputs(outH * outW);
    
            // 각 output cell 계산
            for (size_t i = 0; i < outH; ++i) {
                for (size_t j = 0; j < outW; ++j) {
                    // 2x2 영역의 시작 인덱스
                    size_t base = i * 2 * inputW + j * 2;
    
                    auto sum = ct;
                    // 4개 영역 회전 및 더하기
                    auto rot1 = cc->EvalRotate(ct, 1);
                    auto rot2 = cc->EvalRotate(ct, inputW);
                    auto rot3 = cc->EvalRotate(ct, inputW + 1);
    
                    sum = cc->EvalAdd(sum, rot1);
                    sum = cc->EvalAdd(sum, rot2);
                    sum = cc->EvalAdd(sum, rot3);
    
                    // 해당 영역만 남기도록 mask
                    std::vector<double> mask(slotCount, 0.0);
                    mask[base] = 0.25; // 평균값
                    auto pt_mask = cc->MakeCKKSPackedPlaintext(mask);
    
                    outputs[i * outW + j] = cc->EvalMult(sum, pt_mask);
    
                    // 해당 위치로 앞으로 당김
                    if (base != (i * outW + j)) {
                        int shift = base - (i * outW + j);
                        outputs[i * outW + j] = cc->EvalRotate(outputs[i * outW + j], -shift);
                    }
                }
            }
    
            // 최종 3x3 pack
            Ciphertext<DCRTPoly> packed = cc->EvalAddMany(outputs);
            pooled.push_back(packed);
        }
    
        return pooled;
    }
    
    
    
    // 테스트용 AvgPool 함수 실행
    void TestAvgPool_5x5(CryptoContext<DCRTPoly> cc,
                        const PublicKey<DCRTPoly>& pk,
                        const PrivateKey<DCRTPoly>& sk) {
        size_t inputH = 5, inputW = 5;
        size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

        // ==== 1. 평문 입력 데이터 생성 (1 ~ 25) ====
        std::vector<double> input(slotCount, 0.0);
        for (size_t i = 0; i < inputH; i++) {
            for (size_t j = 0; j < inputW; j++) {
                size_t idx = i * inputW + j;
                input[idx] = (double)(i * inputW + j + 1);
            }
        }

        std::cout << "[Input 5x5]\n";
        for (size_t i = 0; i < inputH; i++) {
            for (size_t j = 0; j < inputW; j++) {
                std::cout << std::setw(4) << input[i * inputW + j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        // ==== 2. 암호화 ====
        Plaintext pt_input = cc->MakeCKKSPackedPlaintext(input);
        Ciphertext<DCRTPoly> ct_input = cc->Encrypt(pk, pt_input);
        

        // ==== 3. 단계적 Rotation Key 생성 ====
        try {

            std::vector<int> rotIndices = {
                1, 5, 6,   // 평균 계산용
                -1, -2, -7, -8, -9, -14, -15, -16  // pack용
            };
            
            
            std::cout << "[추가 Rotation Keys 생성]: ";
            for (auto idx : rotIndices) {
                std::cout << idx << " ";
            }
            std::cout << "\n\n";
            cc->EvalRotateKeyGen(sk, rotIndices);
            
        } catch (const std::exception& e) {
            std::cout << "Rotation key 생성 중 오류: " << e.what() << "\n";
            
            // 오류 발생 시 모든 rotation keys 생성
            std::cout << "[Fallback: 모든 rotation keys 생성]\n";
            std::vector<int> allRotIndices;
            for (int i = -50; i <= 50; i++) {
                if (i != 0) allRotIndices.push_back(i);
            }
            cc->EvalRotateKeyGen(sk, allRotIndices);
        }

       // 4. AvgPooling 실행
    std::vector<Ciphertext<DCRTPoly>> input_channels = { ct_input };
    std::vector<int> poolRotIndices = {1, (int)inputW, (int)inputW + 1};
    cc->EvalRotateKeyGen(sk, poolRotIndices);

    auto pooled = AvgPool2D_MultiChannel_3x3(cc, input_channels, inputH, inputW, pk, poolRotIndices);

    // 5. 복호화 및 3x3 결과 출력
    Plaintext pt_out;
    cc->Decrypt(sk, pooled[0], &pt_out);

    size_t outH = inputH / 2;
    size_t outW = inputW / 2;
    pt_out->SetLength(outH * outW);
    auto vec = pt_out->GetRealPackedValue();

    std::cout << "[AvgPool 결과: 3x3]\n";
    for (size_t i = 0; i < outH; ++i) {
        for (size_t j = 0; j < outW; ++j) {
            size_t idx = i * outW + j;
            std::cout << std::setw(10) << vec[idx] << " ";
        }
        std::cout << "\n";
    }
}


    int main() {
        // 1. CryptoContext 생성
        CCParams<CryptoContextCKKSRNS> params;
        params.SetRingDim(1 << 16);
        params.SetBatchSize(4096);
        params.SetScalingModSize(40);
        params.SetMultiplicativeDepth(5);

        CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
        cc->Enable(PKE);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);

        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);

        // 2. AvgPool 테스트 실행
        TestAvgPool_5x5(cc, keys.publicKey, keys.secretKey);
        

        return 0;
    }