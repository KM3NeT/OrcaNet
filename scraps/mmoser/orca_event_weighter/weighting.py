from __future__ import print_function
import ROOT
import hkkmFluxes
import os, sys
import numpy as np
from math import log10, sqrt, asin, sin
ROOT.gSystem.Load("libOscProb.so")

class OscillationProbability:
    # https://globalfit.astroparticles.es/2018/02/09/empty/
    flvDict = {12: 0, 14:1, 16:2, -12:0, -14:1, -16:2}
    def __init__(self, mh=1):
        #Set parameters to PDG by default
        dm21 = 7.55e-5
        th12 = asin(sqrt(0.320))
        if mh>0: #NO
            dm31 = 2.500e-3
            th13 = asin(sqrt(0.0216))
            th23 = asin(sqrt(0.547))
            dcp = 237.6*ROOT.TMath.Pi()/180.
        else:
            dm31 = -2.42e-3 + dm21
            th13 = asin(sqrt(0.0220))
            th23 = asin(sqrt(0.551))
            dcp = 280.8*ROOT.TMath.Pi()/180.

        # Create PMNS object
        self.myPMNS = ROOT.OscProb.PMNS_Fast()
        # Create default PREM Model
        self.prem = ROOT.OscProb.PremModel()
        self.SetDM21(dm21)
        self.SetDM31(dm31)
        self.SetTheta12(th12)
        self.SetTheta13(th13)
        self.SetTheta23(th23)
        self.SetDeltaCP(dcp)

    def SetDM21(self, dm21):
        self.myPMNS.SetDm(2, dm21)
    def SetDM31(self, dm31):
        self.myPMNS.SetDm(3, dm31)
    def SetTheta12(self, th12):
        self.myPMNS.SetAngle(1, 2, th12)
    def SetTheta13(self, th13):
        self.myPMNS.SetAngle(1, 3, th13)
    def SetTheta23(self, th23):
        self.myPMNS.SetAngle(2, 3, th23)    
    def SetDeltaCP(self, dcp):
        self.myPMNS.SetDelta(1, 3, dcp)
    def PrintOscillationProbabilities(self):
        print("DM21:    ", self.myPMNS.GetDm(2))
        print("DM31:    ", self.myPMNS.GetDm(3))
        print("Theta12: ", self.myPMNS.GetAngle(1, 2))
        print("Theta13: ", self.myPMNS.GetAngle(1, 3))
        print("Theta23: ", self.myPMNS.GetAngle(2, 3))
        print("DeltaCP: ", self.myPMNS.GetDelta(1, 3))


    def GetProbability(self, flavourInitial, flavourFinal, energy, cosz):
        ### flavours +-12 +-14 +- 16 for nue,mu,tau
        flvi = self.flvDict[flavourInitial]
        flvf = self.flvDict[flavourFinal]
        # Skip if cosz unphysical
        if (cosz < -1) or (cosz > 1):
            return -1
      
        # Particles and anti-particles do not mix
        if np.sign(flavourInitial) != np.sign(flavourFinal):
            return 0

        # Set paths in OscProb 
        self.prem.FillPath(cosz)
        self.myPMNS.SetPath(self.prem.GetNuPath())

        # Get probability from OscProb
        self.myPMNS.SetIsNuBar(flavourInitial<0)
        prob = self.myPMNS.Prob(flvi, flvf, energy)
        return prob



oscProb = OscillationProbability()
oscProb.PrintOscillationProbabilities()

hkkm = hkkmFluxes.hkkmFlux()

# nutau
def weightNuTau(energy, cosz):
	if cosz >= 0:
		return 0
	return hkkm.getFlux("numu",energy,cosz)*oscProb.GetProbability(14, 16, energy, cosz)+\
           hkkm.getFlux("nue",energy,cosz)*oscProb.GetProbability(12, 16, energy, cosz)

def weightNuTauBar(energy, cosz):
	if cosz >= 0:
		return 0
	return hkkm.getFlux("anumu",energy,cosz)*oscProb.GetProbability(-14, -16, energy, cosz)+\
           hkkm.getFlux("anue",energy,cosz)*oscProb.GetProbability(-12, -16, energy, cosz)

#numu
def weightNuMu(energy, cosz):
	if cosz >= 0:
		return hkkm.getFlux("numu",energy,cosz)
	return hkkm.getFlux("numu",energy,cosz)*oscProb.GetProbability(14, 14, energy, cosz)+\
           hkkm.getFlux("nue",energy,cosz)*oscProb.GetProbability(12, 14, energy, cosz)
def weightNuMuBar(energy, cosz):
	if cosz >= 0:
		return hkkm.getFlux("anumu",energy,cosz)
	return hkkm.getFlux("anumu",energy,cosz)*oscProb.GetProbability(-14, -14, energy, cosz)+\
           hkkm.getFlux("anue",energy,cosz)*oscProb.GetProbability(-12, -14, energy, cosz)

#nue
def weightNuE(energy, cosz):
	if cosz >= 0:
		return hkkm.getFlux("nue",energy,cosz)
	return hkkm.getFlux("numu",energy,cosz)*oscProb.GetProbability(14, 12, energy, cosz)+\
           hkkm.getFlux("nue",energy,cosz)*oscProb.GetProbability(12, 12, energy, cosz)
def weightNuEBar(energy, cosz):
	if cosz >= 0:
		return hkkm.getFlux("anue",energy,cosz)
	return hkkm.getFlux("anumu",energy,cosz)*oscProb.GetProbability(-14, -12, energy, cosz)+\
           hkkm.getFlux("anue",energy,cosz)*oscProb.GetProbability(-12, -12, energy, cosz)

#NC weights
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


