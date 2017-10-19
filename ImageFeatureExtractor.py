import cv2
import numpy as np
from util import *
from ImageSegmenter import ImageSegmenter

"""
Error when segmentation returns an all black image
"""
class SegmentationError(RuntimeError):
    def __init__(self, method):
        mes = method + " segmentation returned all black image"
        super(SegmentationError, self).__init__(mes)


"""
Class concentrating feature extraction logic
"""

class ImageFeatureExtractor:

    # Hardcoded number of features available
    numfeats = 54
    
    def __init__(self, segmethod):
        self.segmethod = segmethod
        self.segmenter = ImageSegmenter(segmethod)
        self.feats = {}

    """
    extract features
    """

    def extract(self, image, option="list"):
        if type(image) != np.ndarray:
            raise TypeError("ImageFeatureExtractor: not numpy array")

        if image.ndim == 2:
            self.prepare(image)
            return self.calculate(option)

        # interpret as list
        elif image.ndim == 1:
            res = list()
            for i in range(len(image)):
                inst = image[i]
                try:
                    self.prepare(inst)
                except SegmentationError as e:
                    print("index {} filled with 0s: ".format(i) + str(e))
                    res.append(np.zeros(ImageFeatureExtractor.numfeats))
                except Exception as e:
                    print("UNTREATED ERROR AT INDEX {}".format(i))
                    print(str(type(e)) + ": " + str(e))
                    res.append(np.zeros(ImageFeatureExtractor.numfeats))
                else: 
                    res.append(self.calculate(option))
            return res
        else:
            raise TypeError("ImageFeatureExtractor: unexpected ndim")

    """
    set relevant info as instance variables
    """

    def prepare(self, image):
        """
        segment image
        """
        self.segimage = self.segmenter.segment(image)
        if (np.sum(self.segimage != 0) == 0):
            raise SegmentationError(self.segmethod)

        """
        extract contours and relevant info from binary image
        """
        continfo = self.getContourInfo(self.segimage)
        self.segimage = continfo[0]
        self.contours = continfo[1]
        self.hierarchy = continfo[2]
        self.idxMainParticle = getMainContour(self.segimage,
                                              self.contours,
                                              self.hierarchy)
        self.mainContour = self.contours[self.idxMainParticle]
        self.targetArea, self.perimeter = self.getMainContourChars()
        self.totalHoleArea = self.getTotalHoleArea()

        """ get ROI """
        self.ROI, self.brect = self.getROI(image)

        """ get convex hull and characteristics """
        self.hullPoints, self.hull, self.defects = self.getConvexHull()
        self.chArea, self.chPerimeter, self.chCentroid = self.getConvexChars()

        """ min shape info """
        self.rectDiam, self.rectCentroid = self.getMinRect()
        self.circRadius, self.circCentroid = self.getMinCirc()
        self.ellipseDiam, self.ellipseCentroid = self.getMinEllipse()

        """ moments """
        self.mmnts, self.huMmnts, self.logHuMmnts = self.getMoments()

    """
    calculate features assuming prepare has ended
    option determines if dict or list
    if list its returned in processing order
    """

    def calculate(self, option="list"):
        feats = {}
        self.setAreaPerimeterFeats(feats)
        self.setHistMeanStdDevFeats(feats)
        self.setHigherStatisticsFeats(feats)
        self.setMinimalDimension(feats)
        self.setConvexityFeats(feats)
        self.setBoundingShapeFeats(feats)
        self.setShapeFeats(feats)
        self.setHuFeats(feats)
        self.setRelDistFeats(feats)

        if option == "list":
            return list(feats.values())
        elif option == "dict":
            return feats

    def getContourInfo(self, segimage):
        return cv2.findContours(segimage, cv2.RETR_CCOMP,
                                cv2.CHAIN_APPROX_SIMPLE)

    def getMainContourChars(self):
        targetArea = cv2.contourArea(self.mainContour)
        perimeter = cv2.arcLength(self.mainContour, True)
        return targetArea, perimeter

    def getTotalHoleArea(self):
        totalHoleArea = 0
        nextHole = self.hierarchy[0][self.idxMainParticle][2]
        while nextHole != -1:
            totalHoleArea += cv2.contourArea(self.contours[nextHole])
            nextHole = self.hierarchy[0][nextHole][0]
        return totalHoleArea

    def getROI(self, image):
        x, y, w, h = cv2.boundingRect(self.mainContour)
        brect = cv2.rectangle(np.copy(image), (x, y),
                              (x+w, y+h), (0, 255, 0), 2)
        return (x, y, w, h), brect

    def getConvexHull(self):
        hullPoints = cv2.convexHull(self.mainContour)
        hull = cv2.convexHull(self.mainContour, hullPoints, False, False)
        defects = cv2.convexityDefects(self.mainContour, hull.flatten())
        return hullPoints, hull, defects

    def getConvexChars(self):
        chperimeter = cv2.arcLength(self.hullPoints, True)
        charea = cv2.contourArea(self.hullPoints)

        chperimeter = chperimeter if chperimeter > 0 else self.perimeter
        charea = charea if charea > 0 else self.targetArea

        """calculate centroid"""
        chMmnts = cv2.moments(self.hullPoints)
        try:
            centroidx = chMmnts['m10'] / chMmnts['m00']
        except ZeroDivisionError:
            centroidx = 0

        try:
            centroidy = chMmnts['m01'] / chMmnts['m00']
        except ZeroDivisionError:
            centroidy = 0

        chcentroid = (centroidx, centroidy)

        return charea, chperimeter, chcentroid

    def getMinRect(self):
        minRect = cv2.minAreaRect(self.mainContour)
        centroid = minRect[0]
        diam = minRect[1]
        return diam, centroid

    def getMinCirc(self):
        inf = cv2.minEnclosingCircle(self.mainContour)
        centroid = inf[0]
        diam = inf[1]
        return diam, centroid

    def getMinEllipse(self):
        inf = cv2.fitEllipse(self.mainContour)
        centroid = inf[0]
        diam = inf[1]
        return diam, centroid

    def getMoments(self):
        mmnts = cv2.moments(self.mainContour)
        hummnts = cv2.HuMoments(mmnts).flatten()
        loghummnts = np.log(np.abs(hummnts))
        return mmnts, hummnts, loghummnts

    def setAreaPerimeterFeats(self, feats):
        feats['Area_pxl'] = self.targetArea
        feats['Perimeter_pxl'] = self.perimeter
        feats['AreaNoHole_Area_pxl'] = self.targetArea - self.totalHoleArea
        feats['HoleAreaRatio'] = self.totalHoleArea / self.targetArea

    def setHistMeanStdDevFeats(self, feats):
        """ these statistics are calculated with a specific mask """
        """ -1 in drawContours draw them all """
        mask = np.zeros((self.brect.shape[0], self.brect.shape[1]))
        cv2.drawContours(mask, self.contours, -1, 255, 1, 8)
        mask = mask.astype('uint8')

        pmean, stdDevFull = cv2.meanStdDev(self.brect, mask=mask)

        """ take out holes """
        mask = self.segmenter.segment(self.brect) * 255
        meanNoHoles, stdDevNoHoles = cv2.meanStdDev(self.brect, mask=mask)

        """ histogram is a feature """
        hist = np.zeros((8))
        for i in range(self.brect.shape[0]):
            for j in range(self.brect.shape[1]):
                hist[int(self.brect[i][j] // (256 / 8))] += 1

        feats['MeanIntsty_Full'] = pmean[0][0]
        feats['StdDevIntsty_Full'] = stdDevFull[0][0]
        feats['MeanIntsty_noHole'] = meanNoHoles[0][0]
        feats['StdDevIntsty_noHole'] = stdDevNoHoles[0][0]
        feats['Hist32'] = hist[0]
        feats['Hist64'] = hist[1]
        feats['Hist96'] = hist[2]
        feats['Hist128'] = hist[3]
        feats['Hist160'] = hist[4]
        feats['Hist192'] = hist[5]
        feats['Hist224'] = hist[6]
        feats['Hist256'] = hist[7]

    def setHigherStatisticsFeats(self, feats):
        mean = feats['MeanIntsty_Full']
        stddev = feats['StdDevIntsty_Full']
        channels = [0]
        histSize = [256]
        rangesP = [0, 256]
        idxMainParticle = self.idxMainParticle

        """ these statistics are calculated with other mask"""
        mask = np.zeros((self.brect.shape[0], self.brect.shape[1]))
        mask = mask.astype('uint8')
        cv2.drawContours(mask, self.contours, idxMainParticle, 255, 1, 8)

        nextHoleIdx = self.hierarchy[0][idxMainParticle][2]
        while nextHoleIdx != -1:
            cv2.drawContours(mask, self.contours, nextHoleIdx, 0, 1, 8)
            nextHoleIdx = self.hierarchy[0][nextHoleIdx][0]

        althist = cv2.calcHist([self.brect], channels, mask,
                               histSize, rangesP, True, False)
        althist = althist.flatten()

        self.setSkewAndKurt(feats, althist, mean, stddev)
        self.setEntropy(feats, althist)
        self.setMedian(feats, althist)

    def setSkewAndKurt(self, feats, althist, mean, stddev):
        skew = 0
        kurt = 0
        N = np.sum(althist)
        for i in range(len(althist)):
            skew += (althist[i] * (i - mean)) ** 3
            kurt += (althist[i] * (i - mean)) ** 4
        skew = skew / ((N-1) * stddev ** 3)
        kurt = kurt / ((N-1) * stddev ** 4) - 3.0
        feats['Skewness'] = np.log(abs(skew))
        feats['Kurtosis'] = np.log(abs(kurt))

    def setEntropy(self, feats, althist):
        entr = 0
        N = np.sum(althist)
        for i in range(len(althist)):
            binVal = althist[i]
            if binVal == 0:
                continue
            binVal = binVal / N
            binVal = binVal * binVal
            entr -= binVal * np.log10(binVal)

        feats['Entropy'] = entr

    def setMedian(self, feats, althist):
        i = -1
        checkSum = 0
        N = np.sum(althist)
        while checkSum < N/2:
            i += 1
            checkSum += althist[i]
        feats['Median'] = i

    def setMinimalDimension(self, feats):
        w = self.ROI[2]
        h = self.ROI[3]
        mask = np.zeros((h, w), dtype='uint8')
        cv2.drawContours(mask, self.contours, -1, 255, 1, 8)
        mask = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        minDim = 2 * np.max(mask)
        feats['MinorDimInside'] = minDim

    def setConvexityFeats(self, feats):
        solidity = (self.targetArea - self.totalHoleArea) / self.chArea
        convexity = self.chPerimeter / self.perimeter

        maxDef = 0
        if len(self.defects) != 0:
            for defect in self.defects:
                if defect[0][3] > maxDef:
                    maxDef = defect[0][3]

        feats['Max_ConvexityDefect'] = maxDef
        feats['Solidity'] = solidity
        feats['Convexity'] = convexity
        feats['ConvexPerimeter_pxl'] = self.chPerimeter

    def setBoundingShapeFeats(self, feats):
        feats['MinorFeret'] = min(self.rectDiam)
        feats['MajorFeret'] = max(self.rectDiam)

        circarea = 2 * np.pi * self.circRadius ** 2
        feats['MinEnclosingCircleArea_pxl'] = circarea

        ellarea = np.pi * np.prod(self.ellipseDiam) / 4
        feats['ellipsisArea_pxl'] = ellarea

    def setShapeFeats(self, feats):
        targetArea = self.targetArea
        perimeter = self.perimeter

        rectangularity = targetArea / np.prod(self.rectDiam)
        circularity = 4 * np.pi * targetArea / perimeter ** 2
        heywoods = perimeter / (2 * np.sqrt(targetArea * np.pi))
        wadells = 2 * np.sqrt(targetArea / np.pi) / max(self.rectDiam)
        eccentricity = min(self.rectDiam) ** 2 / max(self.rectDiam) ** 2
        elongation = 1.0 - min(self.rectDiam) / max(self.rectDiam)

        compactness = targetArea ** 2
        adjust = 2 * np.pi
        adjust *= np.sqrt(self.mmnts['m20'] ** 2 + self.mmnts['m02'] ** 2)
        compactness = compactness / adjust

        feats['Rectangularity'] = rectangularity
        feats['Circularity'] = circularity
        feats['HeywoodCircularity'] = heywoods
        feats['Wadell_Circularity'] = wadells
        feats['Eccentricity'] = eccentricity
        feats['Elongation'] = elongation
        feats['Compactness'] = compactness

    def setHuFeats(self, feats):
        for i in range(len(self.huMmnts)):
            feats['HU_{:d}'.format(i + 1)] = self.huMmnts[i]
            feats['LogHu_{:d}'.format(i + 1)] = self.logHuMmnts[i]

    def setRelDistFeats(self, feats):
        """Relative distance between centroids"""
        mmnts = self.mmnts
        centroid = (mmnts['m10'] / mmnts['m00'], mmnts['m01'] / mmnts['m00'])
        Dist2BoundBox = calcRelativeDistance(centroid, self.rectCentroid)
        Dist2Ellipse = calcRelativeDistance(centroid, self.ellipseCentroid)
        Dist2Circ = calcRelativeDistance(centroid, self.circCentroid)
        Dist2CHull = calcRelativeDistance(centroid, self.chCentroid)

        feats['RelatDist2BoundingBox'] = Dist2BoundBox
        feats['RelatDist2ConvexHullCtr'] = Dist2CHull
        feats['RelDist2Ellipse'] = Dist2Ellipse
        feats['RelDist2EnclCircle'] = Dist2Circ
