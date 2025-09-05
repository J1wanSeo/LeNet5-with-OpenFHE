// conv_bn_module.cpp
#include "conv_bn_module.h"
#include "relu.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <iostream>


Ciphertext<DCRTPoly> ApproxReLU4_linear(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    return ct_x;
}

Ciphertext<DCRTPoly> ApproxReLU4_square(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    // x^2

    auto x2 = cc->EvalMult(ct_x, ct_x);
    x2 = cc->Rescale(x2);

    // EvalAdd 전에 level 로그 찍기
    // std::cout << "[DEBUG] sum Level: " << sum->GetLevel() << ", pt_const Level: " << pt_const->GetLevel() << std::endl;

    return x2;
}

Ciphertext<DCRTPoly> ApproxReLU4_cryptonet(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    // 0.25 + 0.5 * x + 0.125 * x^2

    auto pt_coeff0   = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.25));
    auto pt_coeff1  = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.5));
    auto pt_coeff2  = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.125));

    auto x1 = cc->EvalMult(ct_x, pt_coeff1);
    x1 = cc->Rescale(x1);
    auto x2_raw = cc->EvalMult(ct_x, ct_x);
    x2_raw = cc->Rescale(x2_raw);
    auto x2 = cc->EvalMult(x2_raw, pt_coeff2);
    x2 = cc->Rescale(x2);

    auto sum = cc->EvalAdd(x1, x2);

    // EvalAdd 전에 level 로그 찍기
    // std::cout << "[DEBUG] sum Level: " << sum->GetLevel() << ", pt_const Level: " << pt_const->GetLevel() << std::endl;

    auto result = cc->EvalAdd(sum, pt_coeff0);
    return result;
}

Ciphertext<DCRTPoly> ApproxReLU4_quad(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize(); 

    // 0.5 * x + 0.204875 * x^2 - 0.0063896 * x^4 + 0.234606

    auto pt_half    = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.5));
    auto pt_coeff2  = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.204875));
    auto pt_coeff4  = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, -0.0063896));
    auto pt_const   = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.234606));

    auto x1 = cc->EvalMult(ct_x, pt_half);
    x1 = cc->Rescale(x1);
    auto x2_raw = cc->EvalMult(ct_x, ct_x);
    x2_raw = cc->Rescale(x2_raw);
    auto x2 = cc->EvalMult(x2_raw, pt_coeff2);
    x2 = cc->Rescale(x2);
    auto x4_raw = cc->EvalMult(x2_raw, x2_raw);
    x4_raw = cc->Rescale(x4_raw);
    auto x4 = cc->EvalMult(x4_raw, pt_coeff4);
    x4 = cc->Rescale(x4);

    auto sum = cc->EvalAdd(x1, x2);
    sum = cc->EvalAdd(sum, x4);

    // EvalAdd 전에 level 로그 찍기
    // std::cout << "[DEBUG] sum Level: " << sum->GetLevel() << ", pt_const Level: " << pt_const->GetLevel() << std::endl;

    auto result = cc->EvalAdd(sum, pt_const);
    return result;
}

Ciphertext<DCRTPoly> ApproxReLU4_Student(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct_x) {
    // size_t slotCount = cc->GetEncodingParams()->GetBatchSize(); 

    //==========================FROM HERE=================================================
    // Insert your own approximation below







    auto result = ct_x; // modify this when implementing your own code







    // Insert your own approximation above
    //=========================TO END=======================================================
    

    return result;
}

// Alpha = 13 다항식 계수 데이터
const std::vector<int> deg_13 = {15, 15, 27};
const std::vector<double> coeff_13 = {
    0.0, 24.558941542500461187, 0.0,
    -669.66044971689436801, 0.0, 6672.9984830133931554,
    0.0, -30603.665616389872425, 0.0,
    73188.403298778778129, 0.0, -94443.321705008449291,
    0.0, 62325.409421254674884, 0.0,
    -16494.674411780599848, 0.0, 9.3562563603543978083,
    0.0, -59.163896393362639749, 0.0,
    148.86093062644842385, 0.0, -175.8128748785829444,
    0.0, 109.11129968595543035, 0.0,
    -36.676883997875556573, 0.0, 6.3184629031129413078,
    0.0, -0.43711341508217764519, 0.0,
    5.078135697588612878, 0.0, -30.732991813718681529,
    0.0, 144.10974681280942417, 0.0,
    -459.66168882614256179, 0.0, 1021.520644704596761,
    0.0, -1620.5625670887702504, 0.0,
    1864.6764641657026581, 0.0, -1567.4930087714349494,
    0.0, 960.9703090934222369, 0.0,
    -424.32616187164667827, 0.0, 131.27850925600366538,
    0.0, -26.9812576626115819, 0.0,
    3.3065138731556502914, 0.0, -0.18274294462753398785
};

/**
 * 다항식 평가 함수 (암호화된 상태에서)
 * @param cc CryptoContext
 * @param ct_x 암호화된 입력
 * @param coefficients 다항식 계수들
 * @return 다항식 계산 결과
 */
Ciphertext<DCRTPoly> EvalPolynomial(CryptoContext<DCRTPoly> cc, 
                                   const Ciphertext<DCRTPoly>& ct_x,
                                   const std::vector<double>& coefficients) {
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    
    // 상수항 (0차항)
    auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, coefficients[0]));
    auto ct_result = ct_x;
    
    if (coefficients.size() > 1) {
        // x의 거듭제곱들을 저장할 벡터
        std::vector<Ciphertext<DCRTPoly>> powers;
        powers.push_back(ct_x); // x^1
        
        // 1차항 계산
        if (std::abs(coefficients[1]) > 1e-15) {
            auto pt_coeff1 = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, coefficients[1]));
            auto term = cc->EvalMult(ct_x, pt_coeff1);
            term = cc->Rescale(term);
            ct_result = cc->EvalAdd(pt, term);
        }
        
        // 2차항 이상 계산
        // #pragma omp parallel for
        for (size_t i = 2; i < coefficients.size(); i++) {
            if (std::abs(coefficients[i]) > 1e-15) {
                // x^i 계산 (필요한 경우만)
                while (powers.size() < i) {
                    auto next_power = cc->EvalMult(powers.back(), ct_x);
                    next_power = cc->Rescale(next_power);
                    powers.push_back(next_power);
                }
                
                // 계수와 곱하기
                auto pt_coeff = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, coefficients[i]));
                auto term = cc->EvalMult(powers[i-1], pt_coeff);
                term = cc->Rescale(term);
                
                // 레벨 맞춤 (필요한 경우)
                while (ct_result->GetLevel() > term->GetLevel()) {
                    ct_result = cc->Rescale(ct_result);
                }
                while (term->GetLevel() > ct_result->GetLevel()) {
                    term = cc->Rescale(term);
                }
                
             ct_result = cc->EvalAdd(ct_result, term);
            }
        }
    }
    
    return ct_result;
}

/**
 * Sign 함수 근사 (암호화된 상태에서)
 * @param cc CryptoContext
 * @param ct_x 암호화된 입력
 * @param B 정규화 상수 (기본값 1.0)
 * @return Sign 함수 근사 결과
 */
Ciphertext<DCRTPoly> ApproxSign(CryptoContext<DCRTPoly> cc, 
                               const Ciphertext<DCRTPoly>& ct_x, 
                               double B = 1.0) {
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    
    // x를 B로 정규화: x_norm = x / B
    auto pt_B_inv = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 1.0 / B));
    auto ct_x_norm = cc->EvalMult(ct_x, pt_B_inv);
    ct_x_norm = cc->Rescale(ct_x_norm);
    
    // 단계별 다항식 적용
    auto ct_current = ct_x_norm;
    size_t coeff_idx = 0;
    
    for (int deg : deg_13) {
        // 현재 단계의 계수들 추출
        std::vector<double> stage_coeffs(coeff_13.begin() + coeff_idx, 
                                        coeff_13.begin() + coeff_idx + deg + 1);
        
        // 다항식 평가
        ct_current = EvalPolynomial(cc, ct_current, stage_coeffs);
        
        coeff_idx += deg + 1;
    }
    
    return ct_current;
}

/**
 * ReLU 근사 함수 (원본 논문 기반)
 * 총 Multiplicative Depth: 약 15-16
 * - Sign 근사: 13 depth (3단계 다항식: 4+4+5)
 * - 정규화: 1 depth (x/B)
 * - ReLU 계산: 2 depth (x * (1+sgn) * 0.5)
 * 
 * @param cc CryptoContext
 * @param ct_x 암호화된 입력
 * @param B 입력 범위 [-B, B] (기본값 1.0)
 * @return 암호화된 ReLU 근사 결과
 */
Ciphertext<DCRTPoly> ApproxReLU_Advanced(CryptoContext<DCRTPoly> cc, 
                                        const Ciphertext<DCRTPoly>& ct_x,
                                        double B = 1.0) {
    size_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    
    // Sign 함수 근사: sgn(x)
    auto ct_sgn = ApproxSign(cc, ct_x, B);
    
    // 1 + sgn(x) 계산
    auto pt_one = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 1.0));
    auto ct_one_plus_sgn = cc->EvalAdd(ct_sgn, pt_one);
    
    // x * (1 + sgn(x)) 계산
    auto ct_x_mult = cc->EvalMult(ct_x, ct_one_plus_sgn);
    ct_x_mult = cc->Rescale(ct_x_mult);
    
    // 최종 결과: x * (1 + sgn(x)) / 2
    auto pt_half = cc->MakeCKKSPackedPlaintext(std::vector<double>(slotCount, 0.5));
    auto result = cc->EvalMult(ct_x_mult, pt_half);
    result = cc->Rescale(result);
    
    return result;
}

std::vector<Ciphertext<DCRTPoly>> ApplyApproxReLU4_All(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    int mode) {

    std::vector<Ciphertext<DCRTPoly>> activated;
    for (auto& ct : ct_channels) {
        // std::cout << "[RELU] Level: " << ct->GetLevel()
        //           << ", Scale: " << ct->GetScalingFactor() << std::endl;

        Ciphertext<DCRTPoly> out;
        if (mode == 0)       out = ApproxReLU4_linear(cc, ct);
        else if (mode == 1)  out = ApproxReLU4_square(cc, ct);
        else if (mode == 2)  out = ApproxReLU4_cryptonet(cc, ct);
        else if (mode == 3)  out = ApproxReLU4_quad(cc, ct);
        else if (mode == 4)  out = ApproxReLU_Advanced(cc, ct);
        else                 out = ApproxReLU4_Student(cc, ct);

        activated.push_back(out);
    }
    return activated;
}

