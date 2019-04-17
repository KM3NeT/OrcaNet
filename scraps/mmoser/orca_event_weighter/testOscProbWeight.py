import ROOT
from math import sqrt, asin, sin
#ROOT.gSystem.Load("/sps/km3net/users/shallman/tauappearance2018/weighting/OscProb/libOscProb.so")
ROOT.gSystem.Load("libOscProb.so")

def GetOscHist(flvi, flvf, mh):
  # Use 200 x bins and 100 y bins
  nbinsx = 200
  nbinsy = 100

  #Set parameters to PDG
  dm21 = 7.5e-5
  th12 = asin(sqrt(0.304))  
  if mh>0:
    dm31 = 2.457e-3
    th13 = asin(sqrt(0.0218))
    th23 = asin(sqrt(0.452))
    dcp = 306*ROOT.TMath.Pi()/180.
  else:
    -2.449e-3 + dm21
  if mh>0:
      th13 = asin(sqrt(0.0219))
      th23 = asin(sqrt(0.579))
      dcp = 254*ROOT.TMath.Pi()/180.

  # Create PMNS object
  myPMNS = ROOT.OscProb.PMNS_Fast()

  # Set PMNS parameters
  myPMNS.SetDm(2, dm21)
  myPMNS.SetDm(3, dm31)
  myPMNS.SetAngle(1,2, th12)
  myPMNS.SetAngle(1,3, th13)
  myPMNS.SetAngle(2,3, th23)
  myPMNS.SetDelta(1,3, dcp)

  # The oscillogram histogram
  h2 = ROOT.TH2D("","",nbinsx,0,50*nbinsx,nbinsy,-1,0)

  # Create default PREM Model
  prem = ROOT.OscProb.PremModel()
  # Loop over cos(theta_z) and L/E
  for ct in range(nbinsy+1):
    # Get cos(theta_z) from bin center
    cosT = h2.GetYaxis().GetBinCenter(ct)

    # Set total path length L  
    L = prem.GetTotalL(cosT)

    # Skip if cosT is unphysical  
    if (cosT < -1) or (cosT > 1):
        continue

    # Fill paths from PREM model
    prem.FillPath(cosT)

    # Set paths in OscProb  
    myPMNS.SetPath(prem.GetNuPath())

    # Loop of L/E  
    for le in range(nbinsx+1):
        # Set L/E from bin center
        loe  = h2.GetXaxis().GetBinCenter(le)

        # Initialize probability
        prob = 0

        # Add probabilities from OscProb
        myPMNS.SetIsNuBar(flvi<0)
        prob = myPMNS.Prob(flvi, flvf, L/loe)


        # Fill probabilities in histogram
        h2.SetBinContent(le,ct,prob)

  # Set titles
  h2.SetTitle(";L/E (km/GeV);cos#theta_{z};P_{#mu#mu} + 0.5#timesP_{#bar{#mu#mu}} + 0.5#timesP_{e#mu} + 0.25#timesP_{#bar{e#mu}}")
  h2.SetDirectory(0)
  return h2

h = GetOscHist(1, 1, 1)
h.Draw("colz")

