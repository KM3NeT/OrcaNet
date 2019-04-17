### Dictionary for run durations from db:
#from runDurationService import *
#RUN_DURATION_DICT = generateRunIDRunDurationDict() #dictionary containing RUNNUMBERS and RUNDURATIONS
###########################################

from icecube import icetray, dataclasses, dataio, antares_common
import numpy as np
from math import exp
from weighting import *




GLOBAL_I3WEIGHTS_HAVEPRINTEDMUONWARNING = False
GLOBAL_I3WEIGHTS_HAVEPRINTEDMUONWARNING2 = False


def oscillatedWeight(object):
		weightFactor = 1.0000760514107537 #from conversion ANTARES year <-> real year
		wcos = -1
		if "I3MCWeightDict" not in object:
				print "no I3MCWeightDict in frame"
				wcos = np.nan
				return wcos
		else:
				#print object["I3MCWeightDict"].get("InteractionProcess")
				neutrinoEnergy = object["I3MCWeightDict"].get("PrimaryNeutrinoEnergy")
				if "OneWeight" not in object["I3MCWeightDict"]:
						global GLOBAL_I3WEIGHTS_HAVEPRINTEDMUONWARNING2
						if not GLOBAL_I3WEIGHTS_HAVEPRINTEDMUONWARNING2:
							print "It seems you have requested to apply COSMIC weight to atmospheric MUPAGE muons. No OneWeight in frame. Returning dummy mupage weight: 3"
							GLOBAL_I3WEIGHTS_HAVEPRINTEDMUONWARNING2 = True
						wcos = 3 #MUPAGE event
						return wcos
				else:
						oneWeight = object["I3MCWeightDict"].get("OneWeight")
						nGenEvents	 = object["I3MCWeightDict"].get("NEvents")
						runNumber	 = object["I3EventHeader"].RunID
						runDuration = 1 #monte carlo without livetime 
						wcos = float(runDuration*oneWeight/nGenEvents*weightFactor)
						primaryParticle = object["AntMCTree"].GetMostEnergeticPrimary()
						pdgParticleID = primaryParticle.GetPdgEncoding()
						#print primaryParticle.GetPdgEncoding(), cos(primaryParticle.zenith), neutrinoEnergy
						
						wcos *= 1e-4 ### factor e-4 from flux unit conversion cm**2 m**2
						if (pdgParticleID) == 12: #nue 
							wcos *= weightNuE(neutrinoEnergy,cos(primaryParticle.zenith)) 
						if (pdgParticleID) == -12: #anue
							wcos *= weightNuEBar(neutrinoEnergy,cos(primaryParticle.zenith))
						if (pdgParticleID) == 14: #numu 
							wcos *= weightNuMu(neutrinoEnergy,cos(primaryParticle.zenith))	   
						if (pdgParticleID) == -14: #anumu
							wcos *= weightNuMuBar(neutrinoEnergy,cos(primaryParticle.zenith))
						if (pdgParticleID) == 16: #nutau 
							wcos *= weightNuTau(neutrinoEnergy,cos(primaryParticle.zenith))		
						if (pdgParticleID) == -16: #anutau
							wcos *= weightNuTauBar(neutrinoEnergy,cos(primaryParticle.zenith))

		return wcos

icetray.I3Frame.oscillatedWeight = oscillatedWeight

def neutralCurrentWeight(object):
        weightFactor = 1.0000760514107537 #from conversion ANTARES year <-> real year
        wcos = -1
        if "I3MCWeightDict" not in object:
                print "no I3MCWeightDict in frame"
                wcos = np.nan
                return wcos
        else:
                #print object["I3MCWeightDict"].get("InteractionProcess")
                neutrinoEnergy = object["I3MCWeightDict"].get("PrimaryNeutrinoEnergy")
                if "OneWeight" not in object["I3MCWeightDict"]:
                        global GLOBAL_I3WEIGHTS_HAVEPRINTEDMUONWARNING2
                        if not GLOBAL_I3WEIGHTS_HAVEPRINTEDMUONWARNING2:
                            print "It seems you have requested to apply COSMIC weight to atmospheric MUPAGE muons. No OneWeight in frame. Returning dummy mupage weight: 3"
                            GLOBAL_I3WEIGHTS_HAVEPRINTEDMUONWARNING2 = True
                        wcos = 3 #MUPAGE event
                        return wcos
                else:
                        oneWeight = object["I3MCWeightDict"].get("OneWeight")
                        nGenEvents   = object["I3MCWeightDict"].get("NEvents")
                        runNumber    = object["I3EventHeader"].RunID
                        runDuration = 1 #monte carlo without livetime 
                        wcos = float(runDuration*oneWeight/nGenEvents*weightFactor)
                        primaryParticle = object["AntMCTree"].GetMostEnergeticPrimary()
                        pdgParticleID = primaryParticle.GetPdgEncoding()
                        #print primaryParticle.GetPdgEncoding(), cos(primaryParticle.zenith), neutrinoEnergy

                        wcos *= 1e-4 ### factor e-4 from flux unit conversion cm**2 m**2
                        if (pdgParticleID) > 0: #nu 
                            wcos *= weightNuNC(neutrinoEnergy,cos(primaryParticle.zenith))
                        if (pdgParticleID) < 0: #anu
                            wcos *= weightNuBarNC(neutrinoEnergy,cos(primaryParticle.zenith))
        return wcos

icetray.I3Frame.neutralCurrentWeight = neutralCurrentWeight

