# OrcaEventWeighter
> Nota bene: This package is still unapproved and experimental

## Setup
Pull and build OscProb (to calculate oscillation probabilities through the Earth) by running
```
source setup.sh
```

... this will also add the OscProb directory to your `PYTHONPATH`

## Files
* hkkmFluxes.py

> reads and provides simple interpolation functions for the HKKM Frejus flux tables

```
    import hkkmFluxes
    # get the default frj-nu-20-01-00 flux table
    hkkm = hkkmFluxes.hkkmFlux()
   
    particle="nue" # others: 'anue', 'numu', 'anumu'
    energy_gev=10.5
    cos_zenith=-0.5
    testFlx = hkkm.getFlux(particle, energy_gev, cos_zenith)
```

* weighting.py
> Provides functions to add the oscillated weight for all neutrino flavours after OscProb probabilities multiplied with the flux from hkkmFluxes.py
```
    import weighting
    pid=-12 #anue
    energy_gev=10.5
    cos_zenith=-0.5
    nueCCweight = weighting.oscillatedWeightCC(energy_gev, cos_zenith, pid)
    neutrinoNCweight = weighting.weightNuNC(energy_gev, cos_zenith)
    antineutrinoNCweight = weighting.weightNuBarNC(energy_gev,cos_zenith)
```    
* oscillatedWeights_PIDoutput.py 

> Provides functions to add the oscillated weight for all neutrino flavours to .h5 after OscProb probabilities multiplied with the HKKM flux.
> You may look into this short script if you want to write weighting functions for other types of files...
