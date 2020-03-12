import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.colors as colors
import os
from operator import itemgetter

print(matplotlib.matplotlib_fname())

plt.rcParams["mathtext.fontset"] = "cm"

"""
def histogram2d(x,y,nx,ny):
	hist,xedges,yedges= np.histogram2d(x,y,[nx,ny],normed=False)
	xe                = 0.5*(xedges[0:xedges.size-1]+xedges[1:xedges.size])
	ye                = 0.5*(yedges[0:yedges.size-1]+yedges[1:yedges.size])
	nbinx             = xe.size
	nbiny             = ye.size
	tothist           = np.sum(hist.astype(float))
	hist              = hist.astype(float)/tothist
	return hist,xe,ye
"""
	
def main():

	# parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
	# parser.add_argument("name", type=str, help="filename (00, 01, 10 or 11)")
	# parser.add_argument("bound", type=str, help="+- bound")
	# parser.add_argument("itercount", type=str, help="iteration count")

	# args = parser.parse_args()
	name = "11"
	# bound = float(args.bound)
	itercount = 10

	files = []
	path = "/Users/nickk124/research/reichart/TRK/TRKrepo/diagnostics/"
	commonname = "TRKpivots" + name
	for i in os.listdir(path):
		if os.path.isfile(os.path.join(path,i)) and commonname in i:
			files.append(i)

	indices = [int(filename[12:14].replace("_","")) for filename in files]

	indexedFiles = {}

	for i, index in enumerate(indices):
		indexedFiles[files[i]] = index

	sortedFilesIndexed = sorted(indexedFiles.items(), key=itemgetter(1))

	sortedFiles = [i[0] for i in sortedFilesIndexed]

	sortedFiles = sortedFiles[0:itercount]

	pivots = []
	plt.figure(num=1,figsize=(10,8),dpi=100,facecolor='white')

	for j in range(0, len(sortedFiles), 6):
		for i in range(0, min(abs(len(sortedFiles)-j), 6)):
			filename = sortedFiles[i+j]

			df = pd.read_csv(filename, sep=" ", header=None)
			ar = df.values
			ar = np.transpose(ar)

			data = ar[0]
			w = ar[1]

			R = data.size
			bincount = np.int(np.sqrt(float(R)))
			
			# plt.figure(num=j,figsize=(12,6),dpi=100,facecolor='white')

			# plt.subplot(2,3,i+1)

			left = -25
			right = 35

			hist, edges, patches = plt.hist(data, bins=bincount, weights = w, facecolor='b', alpha=0.1, range = (left, right), log=True)
			# plt.title("Iteration " + str(i+j))
			plt.ylim(0, 1e5)
			hist, edges = np.histogram(data, bins=bincount, weights = w, range = (left, right))
			plt.xlabel("New Pivot Point " + r'$x_p^{new}$', fontsize=18)
			plt.ylabel("Weighted Count", fontsize=18)
			maxind = np.argmax(hist)
			peakPivot = (edges[maxind + 1] + edges[maxind]) / 2.0

			pivots.append(peakPivot)

			# # Python avg:
			# plt.axvline(x = np.average(data, weights = w), color="k", linestyle="--")
			# # Peak bin val:
			# plt.axvline(x = peakPivot,  color="r", linestyle="-.")


			print("file: %s \t Python average: %f \t peak bin value: %f \t for %i iterations" % (filename, np.average(data, weights = w), peakPivot, itercount) )
		
		# plt.show()
	plt.show()
	avgPivot = np.average(np.array(pivots[1:]))
	print("average of all pivots = %f" % avgPivot)
		
		
		
main()