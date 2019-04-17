import numpy as np
import ROOT
from math import log10,pi,cos,sin
import os, sys
from array import array

class hkkmFlux:
    geniedict = {12: 'nue', -12: 'anue', 14: 'numu', -14: 'anumu', 16: 'nutau', -16: 'anutau'}
    def __init__(self, flux="frj-nu-20-01-000"):
        self.flux = flux
        fluxFile_honda  = os.path.dirname(os.path.realpath(__file__))+"/"+flux+".d"
        fluxFile_graphs = os.path.dirname(os.path.realpath(__file__))+"/"+flux+".root"
        if not os.path.exists(fluxFile_graphs):
            self.convertToGraphs(fluxFile_honda, fluxFile_graphs)
        self.loadGraphs(fluxFile_graphs)

    def convertToGraphs(self, inputFile, outputFile):
        fluxfile = open(inputFile,'r')
        rootfile = ROOT.TFile(outputFile, "RECREATE")
        cosz = -999
        g =   { 'numu' : ROOT.TGraph2D(),
                'anumu': ROOT.TGraph2D(),
                'nue'  : ROOT.TGraph2D(),
                'anue' : ROOT.TGraph2D()  }

        for line in fluxfile:
            line = line.replace(',','').replace('=',' ')
            splitline = line.split()

            if splitline[0] == 'average':
                #print splitline
                cosz = round((float(splitline[6])+float(splitline[4]))/2.,2)
                continue
            elif splitline[0] == 'Enu(GeV)':
                continue
            else:
                energy = float(splitline[0])
                #if energy>200:
                #   continue
                numu = float(splitline[1])
                anumu = float(splitline[2])
                nue = float(splitline[3])
                anue = float(splitline[4])
                g['nue'].SetPoint(g['nue'].GetN(),log10(energy),cosz,nue*energy**2)
                g['numu'].SetPoint(g['numu'].GetN(),log10(energy),cosz,numu*energy**2)
                g['anue'].SetPoint(g['anue'].GetN(),log10(energy),cosz,anue*energy**2)
                g['anumu'].SetPoint(g['anumu'].GetN(),log10(energy),cosz,anumu*energy**2)
                if round(cosz,2)==-0.95:
                    #print cosz
                    thecosz=-1.05
                    g['nue'].SetPoint(g['nue'].GetN(),log10(energy),thecosz,nue*energy**2)
                    g['numu'].SetPoint(g['numu'].GetN(),log10(energy),thecosz,numu*energy**2)
                    g['anue'].SetPoint(g['anue'].GetN(),log10(energy),thecosz,anue*energy**2)
                    g['anumu'].SetPoint(g['anumu'].GetN(),log10(energy),thecosz,anumu*energy**2)
                if round(cosz,2)==0.95:
                    #print cosz
                    thecosz=1.05
                    g['nue'].SetPoint(g['nue'].GetN(),log10(energy),thecosz,nue*energy**2)
                    g['numu'].SetPoint(g['numu'].GetN(),log10(energy),thecosz,numu*energy**2)
                    g['anue'].SetPoint(g['anue'].GetN(),log10(energy),thecosz,anue*energy**2)
                    g['anumu'].SetPoint(g['anumu'].GetN(),log10(energy),thecosz,anumu*energy**2)
        g['nue'].Write('nue')
        g['anue'].Write('anue')
        g['numu'].Write('numu')
        g['anumu'].Write('anumu')
        print("Converted HKKM .d text file to root graphs: " + outputFile)
        rootfile.Close()

    def loadGraphs(self, inputFile):
        rootfile = ROOT.TFile(inputFile, "READONLY")
        self.g = {'nue'  : rootfile.Get('nue'),
                  'anue' : rootfile.Get('anue'),
                  'numu' : rootfile.Get('numu'),
                  'anumu': rootfile.Get('anumu')}
        for it in self.g:
            self.g[it].SetDirectory(0)

    def getFlux(self,fluxtype,energy,cosz):
	    return	self.g[fluxtype].Interpolate(log10(energy),cosz)/energy**2
    def getFluxForIdZcomponent(self,fluxtypenumber,energy,zcomponent):
	    return self.getFlux(self.geniedict[fluxtypenumber],energy,-zcomponent)



def test_exampleFlux():
    ### here some tests:
    hkkm = hkkmFlux()
    print(hkkm.getFluxForIdZcomponent(12,3.6835E+00,0.165970)*1.215E+06)
    ### this is for one test value:
    #neutrino: 1	  107.947		59.266		-13.708    -0.132955	 0.977127	  0.165970	 3.6835E+00		0.000000	 0.968467	  0.162476 1 16 2
    #primarylepton: 1	   107.947		 59.266		 -13.708	-0.332482	  0.941512	   0.054864   3.2613E+00	 0.000000 34
    #target: O16	   14.890
    #track_in:	 1		107.947		  59.266	  -13.708	 -0.059016	   0.770867		0.634256   1.0608E+00	  0.000000 13
    #track_in:	 2		107.947		  59.266	  -13.708	  0.601777	   0.798641    -0.006057   1.1856E+00	  0.000000 14
    #track_in:	 3		107.947		  59.266	  -13.708	 -0.565849	   0.771960    -0.289643   1.6907E+00	  0.000000 4
    #track_in:	 4		107.947		  59.266	  -13.708	  0.153895	   0.739002		0.655890   1.0791E+00	  0.000000 3
    #track_in:	 5		107.947		  59.266	  -13.708	 -0.241334	   0.960527    -0.138366   4.9160E-01	  0.000000 4
    #w2list:	4.716E+15	 3.684E+00	  1.024E-42    6.979E+12   1.0000E+00	 1.000E+00	  6.995E-11
    #weights:	 1.758E+06	  1.215E+06    4.030E+06

def test_fluxRatio_mu2e():
    hkkm = hkkmFlux()
    h = hkkm.g['numu'].GetHistogram()
    h.Add(hkkm.g['anumu'].GetHistogram())
    h.SetTitle('Ratio (a)numu/(a)nue;log10(energy [GeV]);cos(zenith)')
    h2 = hkkm.g['nue'].GetHistogram()
    h2.Add(hkkm.g['anue'].GetHistogram())
    h.Divide(h2)

    stops = [0.00, 0.34, 0.61, 0.84, 1.00]
    red   = [0.00, 0.00, 0.87, 1.00, 0.51]
    green = [0.00, 0.81, 1.00, 0.20, 0.00]
    blue  = [0.51, 1.00, 0.12, 0.00, 0.00]

    s = array('d', stops)
    r = array('d', red)
    g = array('d', green)
    b = array('d', blue)
    ncontours = 100
    npoints = len(s)
    ROOT.TColor.CreateGradientColorTable(npoints, s, r, g, b, ncontours)
    ROOT.gStyle.SetNumberContours(ncontours)
    h.SetMinimum(0.1)
    h.Draw('colz')
    #ROOT.gPad.SetLogz()
    return h

def test_fluxRatio_numu2anumu():
    hkkm = hkkmFlux()
    h = hkkm.g['numu'].GetHistogram()
    h.SetTitle('Ratio numu/anumu;log10(energy [GeV]);cos(zenith)')
    h2 = hkkm.g['anumu'].GetHistogram()
    h.Divide(h2)

    stops = [0.00, 0.34, 0.61, 0.84, 1.00]
    red   = [0.00, 0.00, 0.87, 1.00, 0.51]
    green = [0.00, 0.81, 1.00, 0.20, 0.00]
    blue  = [0.51, 1.00, 0.12, 0.00, 0.00]

    s = array('d', stops)
    r = array('d', red)
    g = array('d', green)
    b = array('d', blue)
    ncontours = 100
    npoints = len(s)
    ROOT.TColor.CreateGradientColorTable(npoints, s, r, g, b, ncontours)
    ROOT.gStyle.SetNumberContours(ncontours)
    h.SetMinimum(0.1)
    h.Draw('colz')
    #ROOT.gPad.SetLogz()
    return h

def test_fluxRatio_nue2anue():
    hkkm = hkkmFlux()
    h = hkkm.g['nue'].GetHistogram()
    h.SetTitle('Ratio nue/anue;log10(energy [GeV]);cos(zenith)')
    h2 = hkkm.g['anue'].GetHistogram()
    h.Divide(h2)

    stops = [0.00, 0.34, 0.61, 0.84, 1.00]
    red   = [0.00, 0.00, 0.87, 1.00, 0.51]
    green = [0.00, 0.81, 1.00, 0.20, 0.00]
    blue  = [0.51, 1.00, 0.12, 0.00, 0.00]

    s = array('d', stops)
    r = array('d', red)
    g = array('d', green)
    b = array('d', blue)
    ncontours = 100
    npoints = len(s)
    ROOT.TColor.CreateGradientColorTable(npoints, s, r, g, b, ncontours)
    ROOT.gStyle.SetNumberContours(ncontours)
    h.SetMinimum(0.1)
    h.Draw('colz')
    #ROOT.gPad.SetLogz()
    return h

def test_fluxRatios1D():
    h = test_fluxRatio_mu2e()
    h.Scale(1./h.GetNbinsY())
    gr = h.ProjectionX().Clone("gr")

    h2 = test_fluxRatio_nue2anue()
    h2.Scale(1./h2.GetNbinsY())
    gr2 = h2.ProjectionX().Clone("gr2")

    h3 = test_fluxRatio_numu2anumu()
    h3.Scale(1./h3.GetNbinsY())
    gr3 = h3.ProjectionX().Clone("gr3")

    gr.SetDirectory(0)
    gr2.SetDirectory(0)
    gr3.SetDirectory(0)
    gr.SetMinimum(0.1)
    gr.Draw('hist')
    gr2.Draw('hist same')
    gr3.Draw('hist same')
    ROOT.gPad.SetLogy()
    ROOT.gPad.Print("flux_ratios.pdf")
#test_fluxRatios1D()

