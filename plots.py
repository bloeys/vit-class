import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def readLossFile(file):
    print('\nReading', file)
    f = open(file)
    lines = f.readlines()
    f.close()
    out = []
    for i in range(min(80000,len(lines))):
        if i%100==0:
            out.append(float(lines[i]))

    return out

def plotFromData(data, title):
    
    #Main plot
    plt.plot(data)
    plt.ylabel('Loss')
    plt.xlabel('Batches')
    plt.ylim(bottom=0, top=0.5)
    plt.title(title)
    matplotlib.pyplot.locator_params(axis='y', nbins=20)

    #Trendline
    x = [i for i in range(len(data))]
    z = np.polyfit(x, data, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    print(title+':', "y=%.6fx+(%.6f)"%(z[0],z[1]))

    plt.show()

if __name__ == '__main__':
    lines = readLossFile('./dense201-stn/trained/original-dn/no-affine/loss-pre-orig-noAffine-noStn-1638672979.2219555.txt')
    print(len(lines))
    plotFromData(lines, 'No Affine No STN')

    lines = readLossFile('./dense201-stn/trained/original-dn/affine/loss-pre-orig-Affine-noStn-1638708176.8370888.txt')
    print(len(lines))
    plotFromData(lines, 'Affine No STN')

    lines = readLossFile('./dense201-stn/trained/original-dn/affine-stn/loss-pre-orig-Affine-Stn-1638766230.8302133.txt')
    print(len(lines))
    plotFromData(lines, 'Affine and STN')
