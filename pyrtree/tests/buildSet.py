#
# Builds Dataset That Maps Point to Index
# Primarily For Training NN
# Author: Logesh Roshan
#

from pyrtree import RTree, Rect
from test_rtree import RectangleGen, TstO, RectWrapper
from testutil import *
import csv
import argparse

parser = argparse.ArgumentParser(description="Build Dataset From RTree")
parser.add_argument('-r', '--rectcount', type=int,
                    default=2000, help="Number of leaf rectangles in RTree")
parser.add_argument('-l', '--length', type=float,
                    default=0.2, help="Max length of leaf rectangles [0,l]")
parser.add_argument('-p', '--points', type=int,
                    default=100, help="Number of points per rectangle")
parser.add_argument('-d', '--dimension', type=int,
                    default=10, help="Number of rectangles per row/col")
parser.add_argument('-o', '--output', type=str,
                    default='NNdataset.csv', help='Output File')
parser.add_argument('-s', '--disjoint', action='store_true',
                    help='Create Disjoint Dataset')
parser.add_argument('-f', '--path', action='store_true',
                    help="Write Full Path to Leaf, for hybrid tree")

args = parser.parse_args()

rgen = RectangleGen()
RECT_COUNT = args.rectcount
MAX_LENGTH = args.length
DATAPOINTS = args.points
DIMENSION = args.dimension
OUTFILE = args.output
DISJOINT = args.disjoint
PATH = args.path

def predefinedTree():
    xs = []
    tree = RTree()
    rid = 0
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            x1 = 10*row
            y1 = 10*col
            x2 = x1 + 5
            y2 = y1 + 5
            rectangle = rgen.rect_from_coords(x1, y1, x2, y2)
            xs.append(RectWrapper(rectangle, rid))
            rid+=1

#    for x in xs:
#        tree.insert(x, x.rect)

    return tree, xs

def newTree():
    xs = [RectWrapper(r, idx) for idx, r in enumerate(take(RECT_COUNT, rgen.rect, MAX_LENGTH))]
    tree = RTree()
    for x in xs:
        tree.insert(x, x.rect)

    return tree, xs

# Prints rectID that Point belongs to
def writeDisjointCSV(pts, opIdx):
    with open(OUTFILE, mode='w') as dset:
        dset_writer = csv.writer(dset, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for pt, op in zip(pts, opIdx):
            branch = list(pt) + [op, ]
            dset_writer.writerow(branch)

# Prints the LeafID that point query produces
def writeLeafIDToCSV(tree, pts, idxMap):
    with open(OUTFILE, mode='w') as dset:
        dset_writer = csv.writer(dset, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for pt in pts:
            path = tree.query_point(pt)
            branch = [child.leaf_obj().idx for child in path if child.is_leaf()]
            branch = list(pt) + branch
            dset_writer.writerow(branch)
#        print(pt, tree.query_point_solutions(pt))

def writePathToCSV(tree, pts, idxMap):
    with open(OUTFILE, mode='w') as dset:
        dset_writer = csv.writer(dset, delimiter=':', quoting=csv.QUOTE_MINIMAL)
        for pt in pts:
            path = tree.query_point_solutions(pt)[0]
#            print(path)
            branch = [idxMap[child] for child in path]
            del branch[0]
            branch = list(pt) + branch
            dset_writer.writerow(branch)
#        print(pt, tree.query_point_solutions(pt))



def createMap(tree_dict):
    idxMap = [0]*len(tree_dict)

    for p, children in tree_dict.items():
        for i,child in enumerate(children):
            idxMap[child] = i

    return idxMap

def boundaryPts(xs):
    ptsIn = []
    for i in range(int(DATAPOINTS/4)):
        for x in xs:
            ptsIn.extend(rgen.pointsOnBoundary(x.rect))

    return ptsIn

def randUniformPts(xs):
    ptsIn = []
    opIdx = []
    for i in range(DATAPOINTS):
        for x in xs:
            ptsIn.append(rgen.pointInside(x.rect))
            opIdx.append(x.idx)

    return ptsIn, opIdx

def main():

    if DISJOINT:
        tree, xs = predefinedTree()
        ptsIn, opIdx = randUniformPts(xs)
        writeDisjointCSV(ptsIn, opIdx)

    else:
        tree, xs = newTree()
        tree_dict = tree.build_dict_index()
        idxMap = createMap(tree_dict)
        ptsIn, _ = randUniformPts(xs)
#        ptsIn = boundaryPts(xs)
        if PATH:
            writePathToCSV(tree, ptsIn, idxMap)
        else:
            writeLeafIDToCSV(tree, ptsIn, idxMap)

if __name__ == "__main__":
    main()
