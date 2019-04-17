import ROOT
from math import log10
from hkkmFluxes import *
import os, sys

### suppression calculation a la Michail
###########################################
suppressionFile = ROOT.TFile(os.path.dirname(os.path.realpath(__file__))+'/RelativeSuppressionHistograms.root', 'READONLY')
h3s = {}
h3s[14] = suppressionFile.Get('h3_numu')
h3s[-14] = suppressionFile.Get('h3_numubar')
h3s[12] = suppressionFile.Get('h3_nue')
h3s[-12] = suppressionFile.Get('h3_nuebar')


def getRelativeSuppression(azi, cosz, energy, fluxtype):
        fluxtype = int(fluxtype)
        if fluxtype not in [-14, -12, 12, 14]:
                print 'fluxtype not allowed, use "14": numu, "-14": anumu, "12": nue or "-12": anue'
                sys.exit()
        #print h3s
        binx = h3s[fluxtype].GetXaxis().FindBin(azi)
        biny = h3s[fluxtype].GetYaxis().FindBin(cosz)
        binz = h3s[fluxtype].GetZaxis().FindBin(log10(energy))

        return h3s[fluxtype].GetBinContent(binx,biny,binz)

##############################################
##############################################

osciFile = ROOT.TFile(os.path.dirname(os.path.realpath(__file__))+'/oscillogramsNH.root','READONLY')
#osci to tau
h_mu2tau = osciFile.Get('NH_nu_mutau')
h_amu2atau = osciFile.Get('NH_nubar_mutau')
h_e2tau = osciFile.Get('NH_nu_etau')
h_ae2atau = osciFile.Get('NH_nubar_etau')
#osci to mu
h_mu2mu = osciFile.Get('NH_nu_mumu')
h_amu2amu = osciFile.Get('NH_nubar_mumu')
h_e2mu = osciFile.Get('NH_nu_emu')
h_ae2amu = osciFile.Get('NH_nubar_emu')
#osci to e
h_mu2e = osciFile.Get('NH_nu_mue')
h_amu2ae = osciFile.Get('NH_nubar_mue')
h_e2e = osciFile.Get('NH_nu_ee')
h_ae2ae = osciFile.Get('NH_nubar_ee')




# nutau
def weightNuTau(energy, cosz, azi):
	if cosz >= 0:
		return 0
	return getFlux("numu",energy,cosz)*getRelativeSuppression(azi,cosz,energy,14)*h_mu2tau.Interpolate(log10(energy),cosz)+getFlux("nue",energy,cosz)*getRelativeSuppression(azi,cosz,energy,12)*h_e2tau.Interpolate(log10(energy),cosz)
def weightNuTauBar(energy, cosz, azi):
	if cosz >= 0:
		return 0
	return getFlux("anumu",energy,cosz)*getRelativeSuppression(azi,cosz,energy,-14)*h_amu2atau.Interpolate(log10(energy),cosz)+getFlux("anue",energy,cosz)*getRelativeSuppression(azi,cosz,energy,-12)*h_ae2atau.Interpolate(log10(energy),cosz)

#numu
def weightNuMu(energy, cosz, azi):
	if cosz >= 0:
		return getFlux("numu",energy,cosz)
	return getFlux("numu",energy,cosz)*getRelativeSuppression(azi,cosz,energy,14)*h_mu2mu.Interpolate(log10(energy),cosz)+getFlux("nue",energy,cosz)*getRelativeSuppression(azi,cosz,energy,12)*h_e2mu.Interpolate(log10(energy),cosz)
def weightNuMuBar(energy, cosz, azi):
	if cosz >= 0:
		return getFlux("anumu",energy,cosz)
	return getFlux("anumu",energy,cosz)*getRelativeSuppression(azi,cosz,energy,-14)*h_amu2amu.Interpolate(log10(energy),cosz)+getFlux("anue",energy,cosz)*getRelativeSuppression(azi,cosz,energy,-12)*h_ae2amu.Interpolate(log10(energy),cosz)

#nue
def weightNuE(energy, cosz, azi):
	if cosz >= 0:
		return getFlux("nue",energy,cosz)
	return getFlux("numu",energy,cosz)*getRelativeSuppression(azi,cosz,energy,14)*h_mu2e.Interpolate(log10(energy),cosz)+getFlux("nue",energy,cosz)*getRelativeSuppression(azi,cosz,energy,12)*h_e2e.Interpolate(log10(energy),cosz)
def weightNuEBar(energy, cosz, azi):
	if cosz >= 0:
		return getFlux("anue",energy,cosz)
	return getFlux("anumu",energy,cosz)*getRelativeSuppression(azi,cosz,energy,-14)*h_amu2ae.Interpolate(log10(energy),cosz)+getFlux("anue",energy,cosz)*getRelativeSuppression(azi,cosz,energy,-12)*h_ae2ae.Interpolate(log10(energy),cosz)

def weightNuNC(energy, cosz, azi):
	return getFlux("nue",energy,cosz)*getRelativeSuppression(azi,cosz,energy,12)+getFlux("numu",energy,cosz)*getRelativeSuppression(azi,cosz,energy,14)
def weightNuBarNC(energy, cosz, azi):
	return getFlux("anue",energy,cosz)*getRelativeSuppression(azi,cosz,energy,-12)+getFlux("anumu",energy,cosz)*getRelativeSuppression(azi,cosz,energy,-14)


#print weightNuTau(10.2,-0.99)
#print weightNuTauBar(10.2,-0.99)


