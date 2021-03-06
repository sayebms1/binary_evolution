#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from functools import reduce
import disk.funcs as dfn
import h5py
import os
import glob
import sys
from matplotlib import pyplot as plt

class binary_mbh(object):    
    def __init__(self, filename):    
        self.parse_file(filename)

    def parse_file(self, filename, cgs_units=True):

        self.filename = filename
        if cgs_units:
            print ('The cgs units are used!')
        with h5py.File(self.filename, 'r') as f:
            self.SubhaloMassInHalfRadType = np.array(f['meta/SubhaloMassInHalfRadType'])
            self.SubhaloSFRinHalfRad = np.array(f['meta/SubhaloSFRinHalfRad'])
            self.snapshot = np.array(f['meta/snapshot'])
            self.subhalo_id = np.array(f['meta/subhalo_id'])
    
            self.masses = np.array(f['evolution/masses'])       #g
            self.mdot = np.array(f['evolution/mdot_eff'])       #g/s
            self.sep       = np.array(f['evolution/sep'])       #cm
            self.dadt      = np.array(f['evolution/dadt'])      #cm/s
            self.dadt_df   = np.array(f['evolution/dadt_df'])   #cm/s
            self.dadt_gw   = np.array(f['evolution/dadt_gw'])   #cm/s
            self.dadt_lc   = np.array(f['evolution/dadt_lc'])   #cm/s
            self.dadt_vd   = np.array(f['evolution/dadt_vd'])   #cm/s
            self.scales    = np.array(f['evolution/scales'])    #NA
            self.times     = np.array(f['evolution/times'])     #s
            self.eccen     = np.array(f['evolution/eccen'])     #NA
            self.z             = (1./self.scales)-1             #NA 
            self.m1 = self.masses[:,0]
            self.m2 = self.masses[:,1]
            self.mtot = self.m1+self.m2
            self.q = self.m2/self.m1
    

    def find_Rlc(self):
        R_lc = np.zeros((self.sep.shape[0],3))
        for i in range(len(self.sep)):
            try:
                idx = reduce(np.intersect1d,(np.where(np.abs(self.dadt_lc[i])>np.abs(self.dadt_df[i]))[0],
                                             np.where(np.abs(self.dadt_lc[i])>np.abs(self.dadt_vd[i]))[0], 
                                             np.where(np.abs(self.dadt_lc[i])>np.abs(self.dadt_gw[i]))[0]))[0]
                R_lc[i]=[i,idx,self.sep[i][idx]]
            except:
                R_lc[i]=[i,np.nan,np.nan]
        return R_lc    
    
    
    
    def find_Rvd(self):
        R_vd = np.zeros((self.sep.shape[0],3))
        for i in range(len(self.sep)):
            try:
                idx = reduce(np.intersect1d,(np.where(np.abs(self.dadt_vd[i])>np.abs(self.dadt_df[i]))[0],
                                             np.where(np.abs(self.dadt_vd[i])>np.abs(self.dadt_lc[i]))[0], 
                                             np.where(np.abs(self.dadt_vd[i])>np.abs(self.dadt_gw[i]))[0]))[0]
                R_vd[i]=[i,idx,self.sep[i][idx]]
            except:
                R_vd[i]=[i,np.nan,np.nan]
        return R_vd
        

    def find_Rgw(self):
        R_gw = np.zeros((self.sep.shape[0],3))
        for i in range(len(self.sep)):
            try:
                idx = reduce(np.intersect1d,(np.where(np.abs(self.dadt_gw[i])>np.abs(self.dadt_df[i]))[0],
                                             np.where(np.abs(self.dadt_gw[i])>np.abs(self.dadt_lc[i]))[0], 
                                             np.where(np.abs(self.dadt_gw[i])>np.abs(self.dadt_vd[i]))[0]))[0]
                R_gw[i]=[i,idx,self.sep[i][idx]]
            except:
                R_gw[i]=[i,np.nan,np.nan]
        return R_gw

    def find_mbin_at_Rvd(self):
        """
        finding mass growth upto disk phase
        """
        R_vd = self.find_Rvd()
        mbin_at_rdisk = np.zeros(self.mtot.size)
        for mm in range(self.mtot.size):
            ti = self.times[mm]
            mdoti = self.mdot[mm]
            if np.isnan(np.sum(R_vd[mm])):
                condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]>np.nanmedian(R_vd[:,-1]))
            else:
                condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]>R_vd[mm][-1])                
                
            ti = ti[condition]
            mdoti = mdoti[condition]
            delta_ti = np.diff(ti)
            mdot_av = 0.5*(mdoti[1:]+mdoti[:-1])
            dmi = mdot_av*delta_ti
            mbin_at_rdisk[mm] = self.mtot[mm] + np.nansum(dmi)
        return mbin_at_rdisk

    def m1m2(self, mbin=None, q=None ):
        if mbin is None:
            mbin = self.mtot
        if q is None:
            q = self.q
        m1 = mbin/(1+q)
        m2 = mbin-m1
        return m1, m2
    
    
    
    def mbin_df_lc(self):
        """
        finding mass growth upto disk phase
        return : an (MxN) matrix of masses for all binaries at all
        separations.
        """
        R_vd = self.find_Rvd()
        mbin_df_lc =-1* np.ones(shape = self.mdot.shape)
        q_df_lc =-1* np.ones(shape = self.mdot.shape)
        m1_df_lc = -1*np.ones(self.mdot.shape)
        m2_df_lc = -1*np.ones(self.mdot.shape)
        #initialize masses and mass ratios from illustris
        mbin_df_lc[:,0] = self.mtot
        q_df_lc[:,0] = self.q
        m1_df_lc[:,0] = self.m1
        m2_df_lc[:,0] = self.m2
        for mm in tqdm(range(self.mtot.size)):
            ti = self.times[mm]
            mdoti = self.mdot[mm]
            if np.isnan(np.sum(R_vd[mm])):
                condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]>=np.nanmedian(R_vd[:,-1]))
            else:
                condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]>=R_vd[mm][-1])   
            q_df_lc[mm][condition] = np.full(q_df_lc[mm][condition].shape, self.q[mm])   #q is not evolving in df_lc
                                                                                         #phase ==> fill with same value
 
            ti = ti[condition]
            mdoti = mdoti[condition]
            delta_ti = np.diff(ti)
            mdot_av = 0.5*(mdoti[1:]+mdoti[:-1])
            dmi = mdot_av*delta_ti            
            idx = np.where(condition)[0]
            for ll in range(len(idx)-1):
                mbin_df_lc[mm][idx[ll]+1] = mbin_df_lc[mm][idx[ll]] + dmi[ll]
                m1_df_lc[mm][idx[ll]+1], m2_df_lc[mm][idx[ll]+1] = self.m1m2(mbin_df_lc[mm][idx[ll]+1]
                                                                             , q_df_lc[mm][idx[ll]+1])
        return m1_df_lc, m2_df_lc, mbin_df_lc, q_df_lc
    
    def find_mrgr_idx(self):
        idx_merged_by_z0     =[]
        idx_not_merged_by_z0 =[]
        for i in range(len(self.z)):
            if 0 in self.z[i]:
                idx_not_merged_by_z0.append(i)
            else:
                idx = np.where(np.isinf(self.z[i]))[0][0]
                idx_merged_by_z0.append(i)
        return np.array(idx_merged_by_z0), np.array(idx_not_merged_by_z0)
    
#     def dm_disk_phase(self):
#         """
#         finds mass growth during disk phase. The inital binary mass in this phase comes
#         from the mass growth in the loss cone and dynamical friction phases.
#         """
#         R_vd = self.find_Rvd()
#         R_gw = self.find_Rgw()
#         m1_after_disk = np.zeros(self.mtot.size)
#         m2_after_disk = np.zeros(self.mtot.size)
#         q_after_disk = -1*np.ones(self.mtot.size)
#         mbin_at_rdisk = self.find_mbin_at_Rvd()
#         for mm in tqdm(range(self.mtot.size)):
            
#             ti = self.times[mm]
#             mdoti = self.mdot[mm]
#             if np.isnan(np.sum(R_vd[mm])):
#                 if np.isnan(np.sum(R_gw[mm])):
#                     print ('this binary has niether a gas dominated phase nor a gw dominated phase')
#                     condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]<=np.nanmedian(R_vd[:,-1]))
#                 else:
#                     condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (R_gw[mm][-1]<self.sep[mm]) & (self.sep[mm]        <=np.nanmedian(R_vd[:,-1]))
                    
#             else:
#                 if np.isnan(np.sum(R_gw[mm])):
#                     #gas dominated all the way
#                     condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]<=R_vd[mm][-1])
#                 else:
#                     condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (R_gw[mm][-1]<self.sep[mm]) & (self.sep[mm]<=R_vd[mm][-1])
        
#             ti = ti[condition]
#             mdoti = mdoti[condition]
#             delta_ti = np.diff(ti)
#             mdot_av = 0.5*(mdoti[1:]+mdoti[:-1])
#             cond_idx = np.where(condition==True)
#             qi = self.q[mm]
#             m1_fin = mbin_at_rdisk[mm]/(1+qi)
#             m2_fin = mbin_at_rdisk[mm]*qi/(1+qi)
#             for jj in range(mdot_av.size):
#                 mdot1, mdot2 = dfn.dm1dm2_lk(qi, mdot_av[jj])
#                 dm1 = mdot1*delta_ti[jj]
#                 dm2 = mdot2*delta_ti[jj]
#                 m1_fin = m1_fin + dm1
#                 m2_fin = m2_fin + dm2
#                 qi = m2_fin/m1_fin
#             m1_after_disk[mm] = m1_fin
#             m2_after_disk[mm] = m2_fin
#             q_after_disk[mm] = qi
            
#         return m1_after_disk, m2_after_disk


#############new function##################    
    def mbin_cbd(self):
        """
        finds mass growth during disk phase. The inital binary mass in this phase comes
        from the mass growth in the loss cone and dynamical friction phases.
        """
        R_vd = self.find_Rvd()
        R_gw = self.find_Rgw()
        #initialize mbin_cbd to mbin_df_lc 
        m1_cbd, m2_cbd, mbin_cbd, q_cbd = self.mbin_df_lc()    
#         print ('shape of m1_cbd {}, m2_cbd {}, mbin_cbd {}, q_cbd{}'.format(m1_cbd.shape, m2_cbd.shape
#                                                                            , mbin_cbd.shape, q_cbd.shape))
        no_condition = 0
        for mm in tqdm(range(self.mtot.size)):
            
            ti = self.times[mm]
            mdoti = self.mdot[mm]
            if np.isnan(np.sum(R_vd[mm])):
                if np.isnan(np.sum(R_gw[mm])):
                    print ('this binary has niether a gas dominated phase nor a gw dominated phase')
                    condition = ((self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) 
                                 & (self.sep[mm]<=np.nanmedian(R_vd[:,-1])))
                else:
                    condition = ((self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]>=R_gw[mm][-1]) 
                                 & (self.sep[mm] <= np.nanmedian(R_vd[:,-1])))
            else:
                if np.isnan(np.sum(R_gw[mm])):
                    #gas dominated all the way
                    condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]<=R_vd[mm][-1])
                else:
                    condition = ((self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]>=R_gw[mm][-1]) 
                                 & (self.sep[mm] <= R_vd[mm][-1]))
            idx = np.where(condition)[0]
            if len(idx)<1:
                no_condition+=1
                print (len(idx))
                print (idx)
                print (no_condition)
            continue
            ti = ti[condition]
            mdoti = mdoti[condition]
            delta_ti = np.diff(ti)
            mdot_av = 0.5*(mdoti[1:]+mdoti[:-1])
            print (q_cbd[mm][idx])

            #print (len(idx), mdot_av.size)
            for ll in range(len(idx)-1):
                if q_cbd[mm][idx[ll]]<0:
                    print ('binary number {} and separation {}'.format(mm, ll))
                    print (q_cbd[mm])
                    print ('')
                mdot1, mdot2 = dfn.dm1dm2_lk(q_cbd[mm][idx[ll]], mdot_av[ll])
                dm1 = mdot1*delta_ti[ll]
                dm2 = mdot2*delta_ti[ll]                
                m1_cbd[mm][idx[ll]+1] = m1_cbd[mm][idx[ll]] + dm1
                m2_cbd[mm][idx[ll]+1] = m2_cbd[mm][idx[ll]] + dm2
                q_cbd[mm][idx[ll]+1] = m2_cbd[mm][idx[ll]+1]/m1_cbd[mm][idx[ll]+1]
                mbin_cbd[mm][idx[ll]+1] = m1_cbd[mm][idx[ll]+1]+m2_cbd[mm][idx[ll]+1]
#                 print ('\n')
#                 print (m1_cbd[mm][idx[ll]], m2_cbd[mm][idx[ll]], mbin_cbd[mm][idx[ll]], q_cbd[mm][idx[ll]])
#                 sys.exit()
            print (q_cbd[mm][idx])
            print ('')
        return m1_cbd, m2_cbd, mbin_cbd, q_cbd

#############End new function##################
    
    
    def mbin_after_insp(self):
        """
        finding mass growth for the whole inspiral
        """
        
        R_vd = self.find_Rvd()
        mbin_after_insp = np.zeros(self.mtot.size)
        for mm in range(self.mtot.size):
            ti = self.times[mm]
            mdoti = self.mdot[mm]
            condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0)
            ti = ti[condition]
            mdoti = mdoti[condition]
            delta_ti = np.diff(ti)
            mdot_av = 0.5*(mdoti[1:]+mdoti[:-1])
            dmi = mdot_av*delta_ti
            dm = np.nansum(dmi)
            mbin_after_insp[mm] = self.mtot[mm] + dm

            
        return mbin_after_insp

    
    
    
class inspiral(object):
    def __init__(self,filename):
        self.spin_magnitudes()
        self.binary_mbh = binary_mbh(filename)
        self.chi1, self.chi2 = self.spin_magnitudes()
    
    def spin_magnitudes(self,use_fgas = True):
        input_dir = '/input/'
        abs_path = os.path.abspath(os.getcwd())
        files= glob.glob('.'+os.path.join(abs_path,input_dir)+'*hdf5')
        fspin = [s for s in files if "spin_magnitude" in s]
        if use_fgas:
            print ("spin magnitudes are gas dependent")
            fspin = [s for s in fspin if "fgas" in s][0]
            print ("result of if", fspin)
        else:
            fspin = [s for s in fspin if "fgas" in s][0]
            print ("spin magnitudes are gas independent")
    
        with h5py.File(fspin,'r') as f:
            primary_dimleesspins   =np.array(f['dimensionlessspins/primary'])
            secondary_dimleesspins =np.array(f['dimensionlessspins/secondary'])
            chi1 = primary_dimleesspins
            chi2 = secondary_dimleesspins
        return chi1, chi2
        
    def modify_dadt_vd(factor=1, mass_growth=False):
        dadt_vd = np.zeros(shape=mdot.shape)
#         m1s = (np.ones(shape=self.binary_mbh.mdot.shape).T*m1).T
#         m2s = (np.ones(shape=self.binary_mbh.mdot.shape).T*m2).T

        if not mass_growth:
            for i in tqdm(range(len(sep))):
                inds = (self.binary_mbh.sep[i]>0.0) 
                dadt_vd[i][inds],d1,regs,d3,d4 = disk_torq.harden(self.binary_mbh.sep[i][inds]
                                                                  , self.binary_mbh.m1[i]
                                                                  , self.binary_mbh.m2[i]
                                                                  , self.binary_mbh.mdot[i][inds]/factor) 
#                 dadt_vd[i][inds],d1,regs,d3,d4 = disk_torq.harden(sep[i][inds],m1s[i][inds],m2s[i][inds],mdot[i][inds]/factor) 
                dadt_vd[i][inds] = np.abs(dadt_vd[i][inds])
        elif mass_growth:
            #substitute the new m1 and m2 masses
                dadt_vd[i][inds],d1,regs,d3,d4 = disk_torq.harden(self.binary_mbh.sep[i][inds]
                                                                  , self.binary_mbh.m1[i]
                                                                  , self.binary_mbh.m2[i]
                                                                  , self.binary_mbh.mdot[i][inds]/factor)
            
            