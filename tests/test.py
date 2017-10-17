import cv2
import numpy as np
from main import *

jupyter0 = {'Area_pxl': 5018.5, 'Perimeter_pxl': 898.933106303215, 'AreaNoHole_Area_pxl': 4414.5, 'HoleAreaRatio': 0.120354687655674, 'MeanIntsty_Full': 115.3704092339979, 'StdDevIntsty_Full': 19.531722670916597, 'MeanIntsty_noHole': 48.849228727815557, 'StdDevIntsty_noHole': 38.278943084259339, 'Hist32': 1892.0, 'Hist64': 2034.0, 'Hist96': 1136.0, 'Hist128': 1089.0, 'Hist160': 1112.0, 'Hist192': 3114.0, 'Hist224': 20356.0, 'Hist256': 1025.0, 'MinorDimInside': 2.0, 'Max_ConvexityDefect': 10502, 'Solidity': 0.3910616999601364, 'Convexity': 0.4608303872931226, 'MinorFeret': 93.91964721679688, 'MajorFeret': 145.78099060058594, 'MinEnclosingCircleArea_pxl': 36586.43401403425, 'ellipsisArea_pxl': 8977.20690528385, 'Skewness': 6.4138664615258696, 'Kurtosis': 11.105800317260233, 'Entropy': 0.08512639231708416, 'Median': 119, 'HU_1': 0.24448411256879604, 'HU_2': 0.0082661960097767002, 'HU_3': 9.7805202167013222e-05, 'HU_4': 4.9339484654135275e-05, 'HU_5': -1.2653967588131319e-09, 'HU_6': 1.7906977992666142e-06, 'HU_7': -3.1853225782977668e-09, 'LogHu_1': -1.4086049514471823, 'LogHu_2': -4.7955808504013344, 'LogHu_3': -9.2325327904463901, 'LogHu_4': -9.9167858917061142, 'LogHu_5': -20.487880120612175, 'LogHu_6': -13.232905182083838, 'LogHu_7': -19.56471227258055, 'Rectangularity': 0.3665359517263509, 'Circularity': 0.07804211710228275, 'HeywoodCircularity': 3.5796080729326412, 'Wadell_Circularity': 0.54832888779760414, 'Eccentricity': 0.41506021931914444, 'Elongation': 0.35574832610295504, 'Compactness': 0.056353299832075369, 'RelatDist2BoundingBox': 7.2876054812498081, 'RelatDist2ConvexHullCtr': 2.7596326196898255, 'RelDist2Ellipse': 4.7347851473221532, 'RelDist2EnclCircle': 14.737693260945852}

jupyter1 = {'AreaNoHole_Area_pxl': 4720.5,
 'Area_pxl': 4744.5,
 'Circularity': 0.5070113185452231,
 'Compactness': 0.0495156401995746,
 'Convexity': 0.7633171453294763,
 'Eccentricity': 0.860242175130472,
 'Elongation': 0.07250758756177844,
 'Entropy': 0.21622082316679925,
 'HU_1': 0.16149638544570288,
 'HU_2': 0.0001313891975564644,
 'HU_3': 3.011549992107595e-05,
 'HU_4': 1.7349242341873418e-09,
 'HU_5': 3.924918301322454e-16,
 'HU_6': 1.9569645266579773e-11,
 'HU_7': 5.670093482927181e-17,
 'HeywoodCircularity': 1.404401137893843,
 'Hist128': 880.0,
 'Hist160': 1017.0,
 'Hist192': 1633.0,
 'Hist224': 6562.0,
 'Hist256': 27241.0,
 'Hist32': 2158.0,
 'Hist64': 1128.0,
 'Hist96': 793.0,
 'HoleAreaRatio': 0.005058488776478028,
 'Kurtosis': 12.153418867682156,
 'LogHu_1': -1.8232725177110458,
 'LogHu_2': -8.937346665670205,
 'LogHu_3': -10.41047057120538,
 'LogHu_4': -20.1723020935574,
 'LogHu_5': -35.474015951836684,
 'LogHu_6': -24.65704146107321,
 'LogHu_7': -37.408740976006285,
 'MajorFeret': 84.38016510009766,
 'Max_ConvexityDefect': 1473,
 'MeanIntsty_Full': 129.94480519480518,
 'MeanIntsty_noHole': 57.32835283528353,
 'Median': 150,
 'MinEnclosingCircleArea_pxl': 12639.657860272528,
 'MinorDimInside': 2.0,
 'MinorFeret': 78.261962890625,
 'Perimeter_pxl': 342.9188275337219,
 'Rectangularity': 0.7184546234374,
 'RelDist2Ellipse': 0.36628494878307366,
 'RelDist2EnclCircle': 1.6660776557628891,
 'RelatDist2BoundingBox': 1.1523168095463048,
 'RelatDist2ConvexHullCtr': 0.32847537393438486,
 'Skewness': 7.686197821436636,
 'Solidity': 0.9114693956362232,
 'StdDevIntsty_Full': 51.79687041416993,
 'StdDevIntsty_noHole': 46.744253312240325,
 'Wadell_Circularity': 0.9211066420851398,
 'ellipsisArea_pxl': 4927.932026306042}

test0 = cv2.imread("../qtTest_shape/p3_frame_289_213_1306.png", 0)
test1 = cv2.imread("../p015_frame_792_1503_141.bmp", 0)
extractor = ImageFeatureExtractor('otsu')
features = extractor.extract(test1, option='dict')

for key in jupyter1.keys():
    if jupyter1[key] == features[key]:
        print(key + " - OK")
    else:
            print(key + " - NOT OK {} vs {}".format(jupyter1[key], features[key]))



