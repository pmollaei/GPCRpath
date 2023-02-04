
import os, glob
import numpy as np
import pandas as pd 
import pandas as pd
import mdtraj as md
from itertools import combinations

# These are the indices of the amino acids in Beta2AR receptor
# Change them if the simulations are not for Beta2AR receptor

res_indx = [(20), (48), (82), (89), (207), (215), (244), (247), (248), (251), (255)] 

res_pair = list(combinations(res_indx, 2))

n = []
for name in sorted(glob.glob("./*lh5")):
    n.append(name)

cont_55_PN_ca_NPY_ang = []
a = []
for name in sorted(glob.glob("./*lh5")):
    trj = md.load_lh5(name, top=None, stride=None, atom_indices=None, frame=None)
    
    features = []
    for frame in range(len(trj)):
        cont_feat = []
        for p in range(len(res_pair)):
            cont = md.compute_contacts(trj[frame], contacts=[res_pair[p]], scheme='ca')
            cont_feat.append(cont[0][0][0])
            
        topo = trj[frame].topology
        res = []
        for rs in topo.residues:
            res.append(rs)    
        atm = []
        for t in topo.atoms:
            atm.append(t)
        res_npxxy = []
        for r in range(len(res)):
            if str(res[r])[:3]=='ASN' and str(res[r+1])[:3]=='PRO' and str(res[r+4])[:3]=='TYR':
                d = (str(res[r])[3:], str(res[r+1])[3:], str(res[r+4])[3:])
                res_npxxy.append(d)
        for ma in range(len(atm)):
            if str(atm[ma]) == 'ASN'+res_npxxy[0][0]+'-'+'O':
                asn_o = (ma)
            if str(atm[ma]) == 'ASN'+res_npxxy[0][0]+'-'+'C':
                asn_c = (ma)
            if str(atm[ma]) == 'ASN'+res_npxxy[0][0]+'-'+'N':
                asn_n = (ma)
        atom_asn = (asn_o, asn_c, asn_n)
        asn_indx = (np.array(atom_asn).reshape(3))
        asn_ = md.compute_angles(trj[frame], [asn_indx], periodic=True, opt=True)
        asn_ang = asn_[0][0]
        
        for mp in range(len(atm)):
            if str(atm[mp]) == 'PRO'+res_npxxy[0][1]+'-'+'O':
                pro_o = (mp)
            if str(atm[mp]) == 'PRO'+res_npxxy[0][1]+'-'+'C':
                pro_c = (mp)
            if str(atm[mp]) == 'PRO'+res_npxxy[0][1]+'-'+'N':
                pro_n = (mp)
        atom_pro = (pro_o, pro_c, pro_n)
        pro_indx = (np.array(atom_pro).reshape(3))
        pro_ = md.compute_angles(trj[frame], [pro_indx], periodic=True, opt=True)
        pro_ang = pro_[0][0]
        
        for mt in range(len(atm)):
            if str(atm[mt]) == 'TYR'+res_npxxy[0][2]+'-'+'O':
                tyr_o = (mt)
            if str(atm[mt]) == 'TYR'+res_npxxy[0][2]+'-'+'C':
                tyr_c = (mt)
            if str(atm[mt]) == 'TYR'+res_npxxy[0][2]+'-'+'N':
                tyr_n = (mt)
        atom_tye = (tyr_o, tyr_c, tyr_n)
        tyr_indx = (np.array(atom_tye).reshape(3))
        tyr_ = md.compute_angles(trj[frame], [tyr_indx], periodic=True, opt=True)
        tyr_ang = tyr_[0][0]
          
        g = (name[52:],frame, asn_ang, pro_ang, tyr_ang, cont_feat)
        features.append(g)
    cont_55_PN_ca_NPY_ang.append(features)
    nn = pd.DataFrame(cont_55_PN_ca_NPY_ang)
    nn.to_pickle('....pkl')
    if len(cont_55_PN_ca_NPY_ang)  % 10 == 0:
        print(len(cont_55_PN_ca_NPY_ang))

