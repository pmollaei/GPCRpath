#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import mdtraj as md
import pandas as pd
import glob
from itertools import combinations


# In[2]:


ntr1 = md.load('template.pdb')
TM1 = ntr1.atom_slice(ntr1.topology.select("resid 10 to 41"))
TM2 = ntr1.atom_slice(ntr1.topology.select("resid 51 to 80"))
TM3 = ntr1.atom_slice(ntr1.topology.select("resid 88 to 122"))
TM4 = ntr1.atom_slice(ntr1.topology.select("resid 132 to 157"))
TM5 = ntr1.atom_slice(ntr1.topology.select("resid 180 to 217"))
TM6 = ntr1.atom_slice(ntr1.topology.select("resid 234 to 267"))
TM7 = ntr1.atom_slice(ntr1.topology.select("resid 274 to 306"))


# In[3]:


ca_TM1 = TM1.xyz[0,TM1.topology.select("name CA"),:]
ca_TM2 = TM2.xyz[0,TM2.topology.select("name CA"),:]
ca_TM3 = TM3.xyz[0,TM3.topology.select("name CA"),:]
ca_TM4 = TM4.xyz[0,TM4.topology.select("name CA"),:]
ca_TM5 = TM5.xyz[0,TM5.topology.select("name CA"),:]
ca_TM6 = TM6.xyz[0,TM6.topology.select("name CA"),:]
ca_TM7 = TM7.xyz[0,TM7.topology.select("name CA"),:]


# In[4]:


def tm_list_gen(ref_TM_inp , check_res):
    ref_TM = [ref_TM_inp]
    check_pdb_TMX_ids = []
    for i in range(len(ref_TM)):
        min_dist = 999
        closest_res = 0
        for j in range(len(check_res)):
            dist_ref_check_ca = np.sqrt(((ref_TM[i][0] - check_res[j][0])**2 +
                                 (ref_TM[i][1] - check_res[j][1])**2 +
                                 (ref_TM[i][2] - check_res[j][2])**2))
            if dist_ref_check_ca < min_dist:
                closest_res = j
                min_dist = dist_ref_check_ca
        check_pdb_TMX_ids.append(closest_res)
    return np.unique(check_pdb_TMX_ids)


# In[5]:


df = pd.read_csv('Labels_GPCRdb.csv',sep=',')
df = np.array(df)


# In[6]:


import itertools
def pair_list_gen(inp):
    pair_list = []
    for pair in itertools.product(inp,inp):
        if pair[0] != pair[1]:
            pair_list.append(tuple(sorted(pair)))
    out_list = list(set([i for i in pair_list]))
    return out_list


# In[7]:


res_name = [ca_TM1[21], ca_TM2[11], ca_TM3[10], ca_TM3[17], ca_TM6[12], ca_TM6[20], ca_TM7[16], ca_TM7[20], ca_TM7[21], ca_TM7[24], ca_TM7[28]]


# In[8]:


tmpairs = list(combinations(res_name, 2))


# In[9]:


angle_incmplt = pd.read_pickle('inact_int_act_NPY_anles_for_incomplete_receptors.pkl')


# In[10]:


cont_incmplt = pd.read_pickle('inact_int_act_55_cont_PN_for_incomplete_receptors.pkl')


# In[11]:


asn_pro_tyr_angles_55_cont_PN = []
for name in glob.glob("TM_only/*.pdb"):
    pdb_ = md.load(name)
    state = (df[np.where(df[:,1] == name[-16:-12])[0],2])
    pdb_id = (df[np.where(df[:,1] == name[-16:-12])[0],1])
    activity = (df[np.where(df[:,1] == name[-16:-12])[0],3])
    contact_list = []
    ASN_angle = []
    PRO_angle = []
    TYR_angle = []
    for tm in range(len(tmpairs)):
        try:
            check_pdb_ca = pdb_.xyz[0,pdb_.topology.select("name CA"),:]
            check_1 = tm_list_gen(tmpairs[tm][0],check_pdb_ca)
            check_2 = tm_list_gen(tmpairs[tm][1],check_pdb_ca)
            check_TM = np.unique(np.concatenate((check_1,check_2)))
            atom_list = np.empty(0).astype(np.int)
            for i in range(len(check_TM)):
                atom_list = np.concatenate((atom_list,pdb_.topology.select('resid '+str(check_TM[i]))))
            res_pairs = pair_list_gen(check_TM)
            cont = md.compute_contacts(pdb_, res_pairs, scheme = "ca")
            contact_list.append(cont[0][0][0])
        except:
            if state==[0]:
                cont0=cont_incmplt.iloc[:,0].tolist()[tm]
                contact_list.append(cont0)
            if state==[1]:
                cont1=cont_incmplt.iloc[:,1].tolist()[tm]
                contact_list.append(cont1)
            if state==[2]:
                cont=cont_incmplt.iloc[:,2].tolist()[tm]
                contact_list.append(cont)
    
    topo = pdb_.topology
    res = []
    for rs in topo.residues:
        res.append(rs)

    atm = []
    for t in topo.atoms:
        atm.append(t)
    
    try:
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
        asn_ = md.compute_angles(pdb_, [asn_indx], periodic=True, opt=True)
        asn_ang = asn_[0][0]
    except:
        if state==[0]:
            asn_ang=angle_incmplt.iloc[:,0].tolist()[0]
        if state==[1]:
            asn_ang=angle_incmplt.iloc[:,1].tolist()[0]
        if state==[2]:
            asn_ang=angle_incmplt.iloc[:,2].tolist()[0]

    try:
        for mp in range(len(atm)):
            if str(atm[mp]) == 'PRO'+res_npxxy[0][1]+'-'+'O':
                pro_o = (mp)
            if str(atm[mp]) == 'PRO'+res_npxxy[0][1]+'-'+'C':
                pro_c = (mp)
            if str(atm[mp]) == 'PRO'+res_npxxy[0][1]+'-'+'N':
                pro_n = (mp)
        atom_pro = (pro_o, pro_c, pro_n)
        pro_indx = (np.array(atom_pro).reshape(3))                   
        pro_ = md.compute_angles(pdb_, [pro_indx], periodic=True, opt=True)
        pro_ang = pro_[0][0]
    except:
        if state==0:
            pro_ang=angle_incmplt.iloc[:,0].tolist()[1]
        if state==1:
            pro_ang=angle_incmplt.iloc[:,1].tolist()[1]
        if state==2:
            pro_ang=angle_incmplt.iloc[:,2].tolist()[1]
            
    try:
        for mt in range(len(atm)):
            if str(atm[mt]) == 'TYR'+res_npxxy[0][2]+'-'+'O':
                tyr_o = (mt)
            if str(atm[mt]) == 'TYR'+res_npxxy[0][2]+'-'+'C':
                tyr_c = (mt)
            if str(atm[mt]) == 'TYR'+res_npxxy[0][2]+'-'+'N':
                tyr_n = (mt)
        atom_tye = (tyr_o, tyr_c, tyr_n)
        tyr_indx = (np.array(atom_tye).reshape(3))                   
        tyr_ = md.compute_angles(pdb_, [tyr_indx], periodic=True, opt=True)
        tyr_ang = tyr_[0][0]
    except:
        if state==0:
            tyr_ang=angle_incmplt.iloc[:,0].tolist()[2]
        if state==1:
            tyr_ang=angle_incmplt.iloc[:,1].tolist()[2]
        if state==2:
            tyr_ang=angle_incmplt.iloc[:,2].tolist()[2]
    data = (pdb_id[0], activity[0], state[0], asn_ang, pro_ang, tyr_ang, contact_list)
    asn_pro_tyr_angles_55_cont_PN.append(data)


# In[12]:


df = pd.DataFrame(asn_pro_tyr_angles_55_cont_PN)


# In[13]:


l = pd.DataFrame(df.iloc[:, -1].tolist())


# In[14]:


result = pd.concat([df.iloc[:, :6], l], axis=1, ignore_index=True)

