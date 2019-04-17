import ROOT
from math import log10
import hkkmFluxes
import os, sys
import numpy as np

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

hkkm = hkkmFlux()

# nutau
def weightNuTau(energy, cosz):
	if cosz >= 0:
		return 0
	return hkkm.getFlux("numu",energy,cosz)*h_mu2tau.Interpolate(log10(energy),cosz)+hkkm.getFlux("nue",energy,cosz)*h_e2tau.Interpolate(log10(energy),cosz)
def weightNuTauBar(energy, cosz):
	if cosz >= 0:
		return 0
	return hkkm.getFlux("anumu",energy,cosz)*h_amu2atau.Interpolate(log10(energy),cosz)+hkkm.getFlux("anue",energy,cosz)*h_ae2atau.Interpolate(log10(energy),cosz)

#numu
def weightNuMu(energy, cosz):
	if cosz >= 0:
		return hkkm.getFlux("numu",energy,cosz)
	return hkkm.getFlux("numu",energy,cosz)*h_mu2mu.Interpolate(log10(energy),cosz)+hkkm.getFlux("nue",energy,cosz)*h_e2mu.Interpolate(log10(energy),cosz)
def weightNuMuBar(energy, cosz):
	if cosz >= 0:
		return hkkm.getFlux("anumu",energy,cosz)
	return hkkm.getFlux("anumu",energy,cosz)*h_amu2amu.Interpolate(log10(energy),cosz)+hkkm.getFlux("anue",energy,cosz)*h_ae2amu.Interpolate(log10(energy),cosz)

#nue
def weightNuE(energy, cosz):
	if cosz >= 0:
		return hkkm.getFlux("nue",energy,cosz)
	return hkkm.getFlux("numu",energy,cosz)*h_mu2e.Interpolate(log10(energy),cosz)+hkkm.getFlux("nue",energy,cosz)*h_e2e.Interpolate(log10(energy),cosz)
def weightNuEBar(energy, cosz):
	if cosz >= 0:
		return hkkm.getFlux("anue",energy,cosz)
	return hkkm.getFlux("anumu",energy,cosz)*h_amu2ae.Interpolate(log10(energy),cosz)+hkkm.getFlux("anue",energy,cosz)*h_ae2ae.Interpolate(log10(energy),cosz)

def weightNuNC(energy, cosz):
	return hkkm.getFlux("nue",energy,cosz)+hkkm.getFlux("numu",energy,cosz)
def weightNuBarNC(energy, cosz):
	return hkkm.getFlux("anue",energy,cosz)+hkkm.getFlux("anumu",energy,cosz)



def oscillatedWeightCC(energy, cosz, pid):
	if pid==-12:
		return weightNuEBar(energy, cosz)
	elif pid==-14:
		return weightNuMuBar(energy, cosz)
	elif pid==-16:
		return weightNuTauBar(energy, cosz)
	elif pid==12:
		return weightNuE(energy, cosz)
	elif pid==14:
		return weightNuMu(energy, cosz)
	elif pid==16:
		return weightNuTau(energy, cosz)
	else:
		sys.exit('Particle id not valid: ' + str(pid))
		return np.nan()
#print weightNuTau(10.2,-0.99)
#print weightNuTauBar(10.2,-0.99)


