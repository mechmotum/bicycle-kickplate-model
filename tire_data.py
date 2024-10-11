from dataclasses import dataclass
from typing import List

# Tire:
# SchwalbeT03: label for Schwalbe 50km Energizer,
# Plus G-Guard 5/Addix-E (28" x 1,75)


@dataclass
class TireCoefficients:
    Fy_coef: List[float]
    Mz_coef: List[float]
    '''Data classes to manage multiple tires, and/or multiple inflation
    pressures'''
    def __str__(self):
        return (f'Tirecoefficients(Fy_coef={self.Fy_coef}, ',
                f'Mz_coef={self.Mz_coef})')


SchwalbeT03_300kPa = TireCoefficients(
    Fy_coef=[
        3.11785071938867,  # MF Coefficients Fy (14 par). For input in deg. Upper Limit Fz: 1000 N
        -752.571579481671,
        1187.69713139730,
        124.477503972100,
        0.760139546489959,
        0.000517867267912517,
        -0.0190283536432610,
        1.48507598866898,
        0.0311484079416778,
        0.0992266045584514,
        -0.0820737048903585,
        7.30337892653729,
        -22.5925899864901,
        5.37222617956484,
    ],
    Mz_coef=[
        2.09906592639809,    # For input in deg. Upper Limit Fz: 1000 N
        10.5876618219117,
        2.60441934159322,
        0.0623247274902716,
        -3.31585328999185,
        -0.758382115661707,
        -0.00823334974409863,
        5.98641545944234,
        -4.75600872857059,
        -0.995097576256860,
        0.0280636613550883,
        0.0706921463627212,
        0.228420786641289,
        -0.178134882851450,
        -0.0205023109359104,
        0.332376760298719,
        0.233694325855722,
        -0.0113037558378034,
    ],
)

SchwalbeT03_400kPa = TireCoefficients(
    Fy_coef=[
        2.41687583414210,  # For input in deg. Upper Limit Fz: 1000 N
        -827.137473764760,
        1237.66083004943,
        395.200360397803,
        3.04468161785720,
        0.00261138053320087,
        -0.922053239766776,
        1.64994486504353,
        0.0379083581395092,
        0.736968117242096,
        -0.266395254094883,
        5.50670683634620,
        -128.332286072010,
        42.3694891796893,
    ],
    Mz_coef=[
        2.21843255129583,  # For input in deg. Upper Limit Fz: 1000 N
        11.5353268949534,
        0.999589243739601,
        -0.421820285221245,
        -4.40274539041674,
        0.189524333398220,
        0.00529677470051710,
        7.18443466280740,
        -16.9117717254093,
        5.14613592375553,
        -0.126298815430486,
        0.0745699176433646,
        -1.21219459523314,
        0.453513932334205,
        0.602307673841529,
        0.0536303220687809,
        -0.656371696645942,
        0.407297064320888,
    ],
)

SchwalbeT03_500kPa = TireCoefficients(
    Fy_coef=[
        2.82190142692177,
        -600.725093731511,
        1126.13419486956,
        108.195059542367,
        0.683427366605219,
        0.00101429526187314,
        -0.885759309085851,
        1.73048558338238,
        0.0486094399802349,
        0.662625384198060,
        -0.175004229143779,
        3.48488836431515,
        59.7995288076727,
        -24.3588719849742,
    ],
    Mz_coef=[
        2.23288298875190,
        5.64395420003480,
        2.85799096871478,
        -16.6499069948331,
        -0.436719975133473,
        1.76980762416801,
        0.0184170389395877,
        -53.2158085410518,
        53.5007295689835,
        -16.1916676969663,
        -0.00987806374096120,
        0.109973767496453,
        -0.0224995998383167,
        0.0595781833578477,
        0.652621022173824,
        0.0958581116776279,
        0.0163525377116817,
        0.0162591874180916
    ]
)
