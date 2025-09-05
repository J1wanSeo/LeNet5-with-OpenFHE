"""
학생용 파일: ReLU 근사 다항식 정의
--------------------------------
이 파일만 수정해서 새로운 activation을 실험하세요!
"""

from .utils_approx import ReLU_maker

# ----------------------------------------------------------
# 학생들이 직접 수정할 수 있는 ReLU 근사 다항식 모음
# ----------------------------------------------------------
quad_relu_polynomials = {
    'linear': (lambda x: x, "x"),
    'square': (lambda x: x ** 2, "x ** 2"),
    'CryptoNet': (lambda x: 0.125 * x**2 + 0.5 * x + 0.25,
                  "0.25 + 0.5 * x + 0.125 * x**2"),
    'quad': (lambda x: 0.234606 + 0.5 * x + 0.204875 * x ** 2 - 0.0063896 * x ** 4,
             "0.234606 + 0.5 * x + 0.204875 * x ** 2 - 0.0063896 * x ** 4"),
    'ReLU-maker': (lambda x: ReLU_maker({'type': 'proposed', 'alpha': 13, 'B': 10})(x),
                   "ReLU Maker with alpha==13"),

    # === 여기에 자신만의 polynomial ReLU를 추가하세요! ==================================
    # Example
    # f(x) = x + x^2로 사용하고 싶은 경우 ->  'student': (lambda x: x + x**2, "x + x^2")로 수정





    'student': (lambda x: x, "insert your own description")








    #=========================================================================
}


# ----------------------------------------------------------
# 유틸 함수
# ----------------------------------------------------------
def get_activation(name: str):

    return quad_relu_polynomials[name][0]

def get_description(name: str):

    return quad_relu_polynomials[name][1]

def list_activations():

    return list(quad_relu_polynomials.keys())
