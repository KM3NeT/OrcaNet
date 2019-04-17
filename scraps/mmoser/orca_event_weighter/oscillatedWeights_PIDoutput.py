from __future__ import print_function
import numpy as np
from math import exp
from weighting import *

def oscillatedWeight(object):
    theWeight = object.weight_one_year

    ### if neutrino: apply oscillated weight
    if bool(object.is_neutrino):
        pdgParticleID = object.type
        neutrinoEnergy = object.energy
        cosz = object.cos_zenith
        if bool(object.is_cc): #CC weight
            if (pdgParticleID) == 12: #nue 
                theWeight *= weightNuE(neutrinoEnergy,cosz) 
            if (pdgParticleID) == -12: #anue
                theWeight *= weightNuEBar(neutrinoEnergy,cosz)
            if (pdgParticleID) == 14: #numu 
                theWeight *= weightNuMu(neutrinoEnergy,cosz)       
            if (pdgParticleID) == -14: #anumu
                theWeight *= weightNuMuBar(neutrinoEnergy,cosz)
            if (pdgParticleID) == 16: #nutau 
                theWeight *= weightNuTau(neutrinoEnergy,cosz)       
            if (pdgParticleID) == -16: #anutau
                theWeight *= weightNuTauBar(neutrinoEnergy,cosz)
        else: #NC weight
            if (pdgParticleID) > 0: #nu 
                theWeight *= weightNuNC(neutrinoEnergy,cosz)
            if (pdgParticleID) < 0: #anu
                theWeight *= weightNuBarNC(neutrinoEnergy,cosz)
    return theWeight

import pandas as pd
def addOscillatedWeightToDataFrame(df, weightColumnName="oscillated_weight_one_year"):
    print("adding oscillated weights to data table:", weightColumnName)
    df[weightColumnName] = df.apply(oscillatedWeight, axis=1)
pd.DataFrame.addOscillatedWeight = addOscillatedWeightToDataFrame


