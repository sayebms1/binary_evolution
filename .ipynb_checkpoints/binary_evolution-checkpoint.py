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
from scipy.interpolate import interp1d

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

    def interpolate_time (self, interp_steps = False):
        from scipy.interpolate import interp1d
        valid_interp_idxs = (self.scales>0)
        f_times = interp1d(self.sep[valid_interp_idxs], self.times[valid_interp_idxs], bounds_error = False, fill_value = -1)
        return f_times
        
#         valid_interp_idxs = (self.scales>0) 
#         x_interp = np.logspace(np.log10(sep_min), np.log10(sep_max), interp_steps)
#         print (x_interp.shape)
#         print (x_interp)
#         interp_sep = x_interp[::-1]
        
#         f_scales = interp1d(self.sep[valid_interp_idxs], self.scales[valid_interp_idxs],bounds_error=False, fill_value=-1)
#         scales = 
        
    


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
  

    def total_mass_growth(self, interp_points, anomalous_q=False):

            my_range = np.array(range(self.mtot.size))
            #initialize the arrays
            mbin_df_lc =-1* np.ones(shape = self.mdot.shape)
            q_df_lc =-1* np.ones(shape = self.mdot.shape)
            m1_df_lc = -1*np.ones(self.mdot.shape)
            m2_df_lc = -1*np.ones(self.mdot.shape)
            #initialize masses and mass ratios from illustris
            mbin_df_lc[:,0] = self.mtot
            q_df_lc[:,0] = self.q
            m1_df_lc[:,0] = self.m1
            m2_df_lc[:,0] = self.m2
            for mm in tqdm(my_range,desc="Calculating total"):
                ti = self.times[mm]
                mdoti = self.mdot[mm]
                condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0)
                #q is irrelevant for total mass change
                q_df_lc[mm][condition] = np.full(q_df_lc[mm][condition].shape, self.q[mm])  
                ti = ti[condition]
                mdoti = mdoti[condition]
                delta_ti = np.diff(ti)
                mdot_av = 0.5*(mdoti[1:]+mdoti[:-1])
                dmi = mdot_av*delta_ti            
                idx = np.where(condition)[0]
                for ll in range(len(idx)-1):
                    mbin_df_lc[mm][idx[ll]+1] = mbin_df_lc[mm][idx[ll]] + dmi[ll]
            return mbin_df_lc


        
        
        
        
        

    
    def mbin_df_lc(self, interp_points, anomalous_q=False):
        """
        finding mass growth upto disk phase
        return : an (MxN) matrix of masses for all binaries at all
        separations.
        """
        sm_2 = 0
        R_vd = self.find_Rvd()
        anom_q = np.array([213, 347, 424, 552, 1026, 1687, 1866, 2385, 3229, 3575, 3792, 4319, 4993, 7096])
        if not anomalous_q:
            tot_range = np.array(range(self.mtot.size))
            my_range = np.setdiff1d(tot_range, anom_q)            

            mbin_df_lc =-1* np.ones(shape = self.mdot.shape)
            q_df_lc =-1* np.ones(shape = self.mdot.shape)
            m1_df_lc = -1*np.ones(self.mdot.shape)
            m2_df_lc = -1*np.ones(self.mdot.shape)
            #initialize masses and mass ratios from illustris
            mbin_df_lc[:,0] = self.mtot
            q_df_lc[:,0] = self.q
            m1_df_lc[:,0] = self.m1
            m2_df_lc[:,0] = self.m2
            for mm in tqdm(my_range,desc="Calculating mass growth in DF,LC stage"):
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
                if len(idx)<=2:
                    print (idx, mm)
                    print ('ti is {}'.format(ti))
                    print ('delta_ti {}'.format(delta_ti) )
                    print ('mdoti is {}'.format(mdoti))
                    print ('mdot_av {}'.format(mdot_av) )
                    print ('dmi is {}'.format(dmi))
                    print ('')
                    sm_2+=1
                for ll in range(len(idx)-1):
                    mbin_df_lc[mm][idx[ll]+1] = mbin_df_lc[mm][idx[ll]] + dmi[ll]
                    m1_df_lc[mm][idx[ll]+1], m2_df_lc[mm][idx[ll]+1] = self.m1m2(mbin_df_lc[mm][idx[ll]+1]
                                                                             , q_df_lc[mm][idx[ll]+1])
            print ('legth smaller than 2 ara this many:{}'.format(sm_2))
            return m1_df_lc, m2_df_lc, mbin_df_lc, q_df_lc
       
        else:
            print ('calculating anomalous q')   
            my_range = np.array([213, 347, 424, 552, 1026, 1687, 1866, 2385, 3229, 3575, 3792, 4319, 4993, 7096])
            x_interp = np.logspace(13, 22, interp_points)
            sep_interp = x_interp[::-1]
            sep_max = self.sep[:,0][my_range]
            
            n_bins = my_range.size
            n_steps = interp_points
            
            m1_df_lc = -1* np.ones(shape = (n_bins, n_steps))
            m2_df_lc = -1* np.ones(shape = (n_bins, n_steps))
            mbin_df_lc = -1* np.ones(shape = (n_bins, n_steps))
            q_df_lc = -1* np.ones(shape = (n_bins, n_steps))
    
            mbin_df_lc[:,0] = self.mtot[my_range]
            q_df_lc[:,0] = self.q[my_range]
            m1_df_lc[:,0] = self.m1[my_range]
            m2_df_lc[:,0] = self.m2[my_range]

            for mm in range(len(my_range)):#tqdm(range(self.mtot.size),desc="Calculating mass growth in DF,LC stage"):
                val_scales = (self.scales[my_range[mm]]>0)
                f_scales = interp1d(self.sep[my_range[mm]][val_scales], self.scales[my_range[mm]][val_scales],bounds_error=False, fill_value=-1)
                f_mdot = interp1d(self.sep[my_range[mm]][val_scales], self.mdot[my_range[mm]][val_scales], bounds_error=False, fill_value=-1)
                f_ti = interp1d(self.sep[my_range[mm]][val_scales], self.times[my_range[mm]][val_scales], bounds_error=False, fill_value=-1)
                #print (sep_max[mm])

                sep_interp_i = np.logspace(13,np.log10(sep_max[mm]), interp_points)[::-1]
                ti = f_ti(sep_interp_i)
                mdoti = f_mdot(sep_interp_i)
                scales = f_scales(sep_interp_i)      
                if np.isnan(np.sum(R_vd[my_range[mm]])):
                    condition = (scales > 0.0) & (scales < 1.0) & (sep_interp_i>=np.nanmedian(R_vd[:,-1]))
                else:
                    condition = (scales > 0.0) & (scales < 1.0) & (sep_interp_i>=R_vd[my_range[mm]][-1])   
                
                #since all qs are gonna stay constan in df lc phase we can fill it up with the same value
                idx = np.where(condition)[0]
                #print (idx)
                
                q_df_lc[mm][condition] = np.full(q_df_lc[mm][condition].shape, self.q[my_range[mm]])   #q is not evolving in df_lc
                                                                                         #phase ==> fill with same value
               # print (q_df_lc[mm][370])
                #sys.exit()
                
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

        
        
        


    def condition (self, stage, scales, sep_interp_i, bin_num):
        """
        Returns the separation indices for each stage based on 
        stage dfinition dadt_stage> all_other_dadt
        """
        R_vd = self.find_Rvd()
        R_gw = self.find_Rgw()
        
        
        stages = ['DF','LC','DF_LC','CBD','GW']
        if stage == stages[0]:
            print ('This is awork in progress')
        if stage == stages[1]:
            print ('Sorry! this one is a work in progress too')
        if stage == stages[2]:
            print ('DF_LC stage')
            if np.isnan(np.sum(R_vd[mm])):
                condition = (scales > 0.0) & (scales < 1.0) & (sep_interp_i>=np.nanmedian(R_vd[:,-1]))
            else:
                condition = (scales > 0.0) & (scales < 1.0) & (sep_interp_i>=R_vd[bin_num][-1]) 
            idx = np.where(condition)[0]
            return idx
        if stage == stages[3]:
            print ('CBD stage ')
            if np.isnan(np.sum(R_vd[my_range[mm]])):
                if np.isnan(np.sum(R_gw[my_range[mm]])):
                    print ('this binary has niether a gas dominated phase nor a gw dominated phase')
                    condition = ((scales > 0.0) & (scales < 1.0)  
                                 & (sep_interp<=np.nanmedian(R_vd[:,-1])))  
                    flag = '1'
                else:
                    omitted+=1
                    print ('This should not happen for some reason')
                    condition = ((scales > 0.0) & (scales < 1.0) & (sep_interp_i>=R_gw[my_range[mm]][-1]) 
                                    & (sep_interp_i <= np.nanmedian(R_vd[:,-1])))
                    flag = '2'
            else:
                if np.isnan(np.sum(R_gw[my_range[mm]])):
                    condition = (scales > 0.0) & (scales < 1.0) & (sep_interp_i<=R_vd[my_range[mm]][-1])
                    flag = '3'
    
                else:
                    condition = ((scales > 0.0) & (scales < 1.0) & (sep_interp_i>=R_gw[my_range[mm]][-1]) 
                                     & (sep_interp_i <= R_vd[my_range[mm]][-1]))
                    flag = '4'
            idx = np.where(condition)[0]
            return idx
            
            
            
            
    
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
    
    def dm_disk_phase(self):
        """
        finds mass growth during disk phase. The inital binary mass in this phase comes
        from the mass growth in the loss cone and dynamical friction phases.
        """
        R_vd = self.find_Rvd()
        R_gw = self.find_Rgw()
        m1_after_disk = np.zeros(self.mtot.size)
        m2_after_disk = np.zeros(self.mtot.size)
        q_after_disk = -1*np.ones(self.mtot.size)
        mbin_at_rdisk = self.find_mbin_at_Rvd()
        for mm in tqdm(range(self.mtot.size)):
            
            ti = self.times[mm]
            mdoti = self.mdot[mm]
            if np.isnan(np.sum(R_vd[mm])):
                if np.isnan(np.sum(R_gw[mm])):
                    print ('this binary has niether a gas dominated phase nor a gw dominated phase')
                    condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]<=np.nanmedian(R_vd[:,-1]))
                else:
                    condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (R_gw[mm][-1]<self.sep[mm]) & (self.sep[mm]        <=np.nanmedian(R_vd[:,-1]))
                    
            else:
                if np.isnan(np.sum(R_gw[mm])):
                    #gas dominated all the way
                    condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]<=R_vd[mm][-1])
                else:
                    condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (R_gw[mm][-1]<self.sep[mm]) & (self.sep[mm]<=R_vd[mm][-1])
        
            ti = ti[condition]
            mdoti = mdoti[condition]
            delta_ti = np.diff(ti)
            mdot_av = 0.5*(mdoti[1:]+mdoti[:-1])
            cond_idx = np.where(condition==True)
            qi = self.q[mm]
            m1_fin = mbin_at_rdisk[mm]/(1+qi)
            m2_fin = mbin_at_rdisk[mm]*qi/(1+qi)
            for jj in range(mdot_av.size):
                mdot1, mdot2 = dfn.dm1dm2_lk(qi, mdot_av[jj])
                dm1 = mdot1*delta_ti[jj]
                dm2 = mdot2*delta_ti[jj]
                m1_fin = m1_fin + dm1
                m2_fin = m2_fin + dm2
                qi = m2_fin/m1_fin
            m1_after_disk[mm] = m1_fin
            m2_after_disk[mm] = m2_fin
            q_after_disk[mm] = qi
            
        return m1_after_disk, m2_after_disk


#############new functions##################    

    def L_edd(self,m):
        '''returns L_edd in cgs units'''
        factor= (4*np.pi*astc.G*astc.MP*astc.C)/(astc.THOMSON)
        return factor*m
    def Mdot_edd(self,m):
        epsilon=0.2
        return L_edd(m)/(epsilon*astc.C**2)

    def mbin_cbd(self, interp_points, anomalous_q=False):
        """
        finds mass growth during disk phase. The inital binary mass in this phase comes
        from the mass growth in the loss cone and dynamical friction phases.
        """
        anom_q = np.array([213, 347, 424, 552, 1026, 1687, 1866, 2385, 3229, 3575, 3792, 4319, 4993, 7096])
        if anomalous_q:
            my_range = anom_q
            print ('calculating anomalous q')
            x_interp = np.logspace(13, 22, interp_points)             
            sep_interp = x_interp[::-1]
            sep_max = self.sep[:,0][my_range]
            print ('performing interpolation with {} points'.format(interp_points))
            R_vd = self.find_Rvd()
            R_gw = self.find_Rgw()
            #interpolate all m1, m2, mbin, and q to go with the 
            print ('anomalous_q {}, interp_points {}'.format(anomalous_q, interp_points))
            m1_df_lc, m2_df_lc, mbin_df_lc, q_df_lc = self.mbin_df_lc(interp_points, anomalous_q) 
            m1_cbd, m2_cbd, mbin_cbd, q_cbd = self.mbin_df_lc(interp_points, anomalous_q)
            print (q_df_lc[0][370])

         
            no_cond = 0
            omitted = 0            
            
            for mm in range(len(my_range)):
                print ('\n\nThis is binary {}'.format(mm))
                val_scales = (self.scales[my_range[mm]]>0)
                f_scales = interp1d(self.sep[my_range[mm]][val_scales], self.scales[my_range[mm]][val_scales],bounds_error=False, fill_value=-1)
                f_mdot = interp1d(self.sep[my_range[mm]][val_scales], self.mdot[my_range[mm]][val_scales], bounds_error=False, fill_value=-1)
                f_ti = interp1d(self.sep[my_range[mm]][val_scales], self.times[my_range[mm]][val_scales], bounds_error=False, fill_value=-1)

                sep_interp_i = np.logspace(13,np.log10(sep_max[mm]), interp_points)[::-1]
                ti = f_ti(sep_interp_i)
                mdoti = f_mdot(sep_interp_i)
                scales = f_scales(sep_interp_i) 
                
#                 ti = f_ti(sep_interp)
#                 mdoti = f_mdot(sep_interp)
#                 scales = f_scales(sep_interp)
                if np.all(self.dadt_vd[my_range[mm]] == 0):
                    omitted+=1
                    print ('dadt_vd=0 for all separation')

                if np.isnan(np.sum(R_vd[my_range[mm]])):
                    if np.isnan(np.sum(R_gw[my_range[mm]])):
                        print ('this binary has niether a gas dominated phase nor a gw dominated phase')
                        condition = ((scales > 0.0) & (scales < 1.0) 
                                     & (sep_interp<=np.nanmedian(R_vd[:,-1])))  
                        flag = '1'
                    else:
                        omitted+=1
                        print ('This should not happen for some reason')
                        condition = ((scales > 0.0) & (scales < 1.0) & (sep_interp_i>=R_gw[my_range[mm]][-1]) 
                                     & (sep_interp_i <= np.nanmedian(R_vd[:,-1])))
                        flag = '2'
                else:
                    if np.isnan(np.sum(R_gw[my_range[mm]])):
                        condition = (scales > 0.0) & (scales < 1.0) & (sep_interp_i<=R_vd[my_range[mm]][-1])
                        flag = '3'
    
                    else:
                        condition = ((scales > 0.0) & (scales < 1.0) & (sep_interp_i>=R_gw[my_range[mm]][-1]) 
                                     & (sep_interp_i <= R_vd[my_range[mm]][-1]))
                        flag = '4'
            

                idx = np.where(condition)[0]
                if len(idx)<2:
                    print ('length is less than 2 ==> no good')
                    omitted+=1
                    no_cond+=1
                else:
                    ti = ti[condition]
                    mdoti = mdoti[condition]
                    delta_ti = np.diff(ti)
                    mdot_av = 0.5*(mdoti[1:]+mdoti[:-1])
                    for ll in range(len(idx)-1):
                        if ll == 0:
                            q_cbd[mm][idx[ll]] = q_df_lc[mm][idx[ll]-1]
                            m1_cbd[mm][idx[ll]] = m1_df_lc[mm][idx[ll]-1] 
                            m2_cbd[mm][idx[ll]] = m2_df_lc[mm][idx[ll]-1]
                            mbin_cbd[mm][idx[ll]] = mbin_df_lc[mm][idx[ll]-1]
#                             if mm ==3:
#                                 print ('in the loop after initial assignment')
#                                 print ('q_df_lc[mm] {}'.format(q_df_lc[mm]))
#                                 print ('q_cbd[mm][idx] {}'.format(q_cbd[mm][idx]))
#                                 print ('m1_cbd[mm][idx] {}'.format(m1_cbd[mm][idx]))
#                                 print ('m2_cbd[mm][idx] {}'.format(m2_cbd[mm][idx]))                            
#                                 print ('mbin_cbd[mm][idx] {}'.format(mbin_cbd[mm][idx]))                            
                        if q_cbd[mm][idx[ll]]<=1:
                            #print ('q_cbd inside of q<=1 is: {}'.format(q_cbd[mm][idx[ll]]))
                            mdot1, mdot2 = dfn.dm1dm2_lk(q_cbd[mm][idx[ll]], mdot_av[ll])
                            
                            dm1 = mdot1*delta_ti[ll]
                            dm2 = mdot2*delta_ti[ll]      
#                             if mm ==3: 
#                                 print ('mdoti[ll] {}'.format(mdoti[ll]))
#                                 print ('mdoti[ll] {}'.format(mdoti[ll+1]))
#                                 print ('mdot_av[ll]] {}'.format(mdot_av[ll]))
#                                 print ('mdot1: {}, mdot2: {}'.format(mdot1, mdot2))
#                                 print ('delta_ti {:e}'.format(delta_ti[ll]))
#                                 print ('dm1: {:e}, dm2: {:e}'.format(dm1, dm2))
#                                 print ('dm1/m1= {}, dm2/m2= {}'.format(dm1/m1_cbd[mm][idx[ll]], dm2/m2_cbd[mm][idx[ll]]))
                            
                            m1_cbd[mm][idx[ll+1]] = m1_cbd[mm][idx[ll]] + dm1
                            m2_cbd[mm][idx[ll+1]] = m2_cbd[mm][idx[ll]] + dm2
                            q_cbd[mm][idx[ll+1]] = m2_cbd[mm][idx[ll]+1]/m1_cbd[mm][idx[ll]+1]  
#                             print (q_cbd[mm][idx[ll]])
#                             print ('\n\nq_cbd[mm]', q_cbd[mm][370])
#                             print ('q_df_lc is', q_df_lc[mm][369])
#                             print ('This is q',q_cbd[mm][idx[ll]])
#                             print ('This is q[ll+1]',q_cbd[mm][idx[ll+1]])
#                             print ('this binary nubmer is {} and bin index is {}'.format(mm, my_range[mm] ))
#                             sys.exit()
                            
                        elif q_cbd[mm][idx[ll]]>1:

                            print ('q>1 and the values is {} the previous step q:{}'.format(q_cbd[mm][idx[ll]],q_cbd[mm][idx[ll]-1] ))
                            print ('binary {} and index {}'.format(mm,ll))
                            print (q_cbd[mm])
                            print (idx)
                            sys.exit()
                            tmp = q_cbd[mm][idx[ll]]
                            q_cbd[mm][idx[ll]] = 1/tmp
                            mdot2, mdot1 = dfn.dm1dm2_lk(q_cbd[mm][idx[ll]], mdot_av[ll])
                            dm1 = mdot1*delta_ti[ll]
                            dm2 = mdot2*delta_ti[ll]                
                            m1_cbd[mm][idx[ll+1]] = m1_cbd[mm][idx[ll]] + dm1
                            m2_cbd[mm][idx[ll+1]] = m2_cbd[mm][idx[ll]] + dm2
                            q_cbd[mm][idx[ll+1]] = m1_cbd[mm][idx[ll]+1]/m2_cbd[mm][idx[ll]+1]
                            print (q_cbd[mm][idx[ll]])
                            #print ('delta_ti in cm is {}'.format())
                            print ('')
                            sys.exit()
            return m1_cbd, m2_cbd, mbin_cbd, q_cbd

        elif not anomalous_q:
            print ('calculating non anomalous q')
            tot_range = np.array(range(self.mtot.size))
            my_range = np.setdiff1d(tot_range, anom_q)
            #The later if statements gets rid of the anomalous qs
            R_vd = self.find_Rvd()
            R_gw = self.find_Rgw()
            #initialize mbin_cbd to mbin_df_lc #initalization happening here becuase for anomalous q after interplation the dimensionsa re different
            m1_cbd, m2_cbd, mbin_cbd, q_cbd = self.mbin_df_lc(interp_points, False)    
            
            no_cond = 0
            omitted = 0
        
            for mm in tqdm(my_range,desc="Calculating mass growth in CBD stage"):
                ti = self.times[mm]
                mdoti = self.mdot[mm]
                if np.all(self.dadt_vd[mm] == 0):
                    omitted+=1
                    continue
                if np.isnan(np.sum(R_vd[mm])):
                    if np.isnan(np.sum(R_gw[mm])):
                        print ('this binary has niether a gas dominated phase nor a gw dominated phase')
                        condition = ((self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) 
                                     & (self.sep[mm]<=np.nanmedian(R_vd[:,-1])))  
                        flag = '1'
                    else:
                        omitted+=1
                        continue
                        condition = ((self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]>=R_gw[mm][-1]) 
                                     & (self.sep[mm] <= np.nanmedian(R_vd[:,-1])))
                        flag = '2'
                else:
                    if np.isnan(np.sum(R_gw[mm])):
                        condition = (self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]<=R_vd[mm][-1])
                        flag = '3'
    
                    else:
                        condition = ((self.scales[mm] > 0.0) & (self.scales[mm] < 1.0) & (self.sep[mm]>=R_gw[mm][-1]) 
                                     & (self.sep[mm] <= R_vd[mm][-1]))
                        flag = '4'
            
                idx = np.where(condition)[0]
            
                if len(idx)<2:
                    omitted+=1
                    no_cond+=1
                    continue
                else:
                    ti = ti[condition]
                    mdoti = mdoti[condition]
                    delta_ti = np.diff(ti)
                    mdot_av = 0.5*(mdoti[1:]+mdoti[:-1])
                    for ll in range(len(idx)-1):
                        if q_cbd[mm][idx[ll]]<=1:
                            mdot1, mdot2 = dfn.dm1dm2_lk(q_cbd[mm][idx[ll]], mdot_av[ll])
                            dm1 = mdot1*delta_ti[ll]
                            dm2 = mdot2*delta_ti[ll]      
                            m1_cbd[mm][idx[ll+1]] = m1_cbd[mm][idx[ll]] + dm1
                            m2_cbd[mm][idx[ll+1]] = m2_cbd[mm][idx[ll]] + dm2
                            q_cbd[mm][idx[ll+1]] = m2_cbd[mm][idx[ll]+1]/m1_cbd[mm][idx[ll]+1]
#                             print ('\n\n q is smaller than 1 ')
#                             print ('mdot1:',mdot1)
#                             print ('mdot2',mdot2)
#                             print ('m1[ll]',m1_cbd[mm][idx[ll]])
#                             print ('m1[ll]+1',m1_cbd[mm][idx[ll]+1])
#                             print ('m1[ll+1]',m1_cbd[mm][idx[ll+1]])
#                             print ('m2[ll]',m2_cbd[mm][idx[ll]])
#                             print ('m2[[ll]+1]',m2_cbd[mm][idx[ll]+1])
#                             print ('m2[ll+1]',m2_cbd[mm][idx[ll+1]])
#                             print ('dm2/m2[ll]',dm2/m2_cbd[mm][idx[ll]] )
#                             print ('dm1/m1[ll]',dm1/m1_cbd[mm][idx[ll]] )
#                             print ()
    
                            
#                             print ('delta_ti[ll]',delta_ti[ll])
                            
                        elif q_cbd[mm][idx[ll]]>1:
    #                         print (q_cbd[mm])
#                             print (mm)
                            print ('q>1 and the values is {} the previous step q:{}'.format(q_cbd[mm][idx[ll]],q_cbd[mm][idx[ll]-1] ))
                            tmp = q_cbd[mm][idx[ll]]
                            q_cbd[mm][idx[ll]] = 1/tmp
                            mdot2, mdot1 = dfn.dm1dm2_lk(q_cbd[mm][idx[ll]], mdot_av[ll])
                            dm1 = mdot1*delta_ti[ll]
                            dm2 = mdot2*delta_ti[ll]                
                            m1_cbd[mm][idx[ll+1]] = m1_cbd[mm][idx[ll]] + dm1
                            m2_cbd[mm][idx[ll+1]] = m2_cbd[mm][idx[ll]] + dm2
                            q_cbd[mm][idx[ll+1]] = m1_cbd[mm][idx[ll]+1]/m2_cbd[mm][idx[ll]+1]
                            print (q_cbd[mm][idx[ll]])
#                             print ('delta_ti in cm is {}'.format())
#                             print ('')
                            sys.exit()

            print (no_cond)
            print (omitted)
            return m1_cbd, m2_cbd, mbin_cbd, q_cbd
        else:
            print ('The flag can either be True of False')
            print ('Exiting the program')
            sys.exit()
        print (no_cond)
        print (omitted)        
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