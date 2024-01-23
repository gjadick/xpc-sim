#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:38:14 2024

@author: giavanna
"""

import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path + '/../..')

#%% update rcParams

plt.rcParams.update({
    'figure.dpi': 300,
    'font.size':10,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': True,
    'axes.titlesize':10,
    'axes.labelsize':8,
    'axes.linewidth': .5,
    'xtick.top': True, 
    'ytick.right': True, 
    'xtick.direction': 'in', 
    'ytick.direction': 'in',
    'xtick.labelsize':8,
    'ytick.labelsize':8,
     'legend.fontsize': 8,
     #'lines.markersize':5,
     'lines.linewidth':1,
     })


    
def getfile(fname, Nx, Ny=None, dtype=np.float32):
    if Ny is None:
        Ny = Nx
    return np.fromfile(fname, dtype=dtype).reshape([Ny, Nx])


def absdiff(M1, M2):
    return np.abs(M1 - M2)
    
if __name__ == '__main__':
    
        
    FSZ = 12
    txtSZ = 8
    Ntxt = 40
    
    
    propdists = [1e-3, 50e-3, 100e-3]
    recon_sizes = [512, 2048, 4096]
    FOV, ramp = 5e-3, 1.0
    N_slices = 10
    #paramfiles = ['params_tissue_highres.txt', 'params_tissue_2um.txt', 'params_highres.txt', 'params_2um.txt']
    #runs = ['tissuefish_highres', 'tissuefish_2um', 'metalfish_highres', 'metalfish_2um']
    runs = ['tissuefish_highres', 'tissuefish_2um']
    outdir = 'output/ct24/'
    figdir = outdir + 'figs/'
    
    # do_crop = False
    # if do_crop: 
    #     x0 = int(0.2*N_matrix)
    #     y0 = int(0.35*N_matrix)
    #     N_crop = int(0.3*N_matrix)
    #     T_recon_prj = T_recon_prj[y0:y0+N_crop, x0:x0+N_crop]
    #     T_recon_ms = T_recon_ms[y0:y0+N_crop, x0:x0+N_crop]
        
    # #%% read the data
    recons_prj = {}
    recons_ms = {}    
    pagrecons_prj = {}
    pagrecons_ms = {}

    for run_id in runs:
        recons_prj[run_id] = {}
        recons_ms[run_id] = {}
        pagrecons_prj[run_id] = {}
        pagrecons_ms[run_id] = {}
        
        for R in propdists:
            recons_prj[run_id][R] = {}
            recons_ms[run_id][R] = {}
            pagrecons_prj[run_id][R] = {}
            pagrecons_ms[run_id][R] = {}
            
            imdir = outdir + run_id 
            for N_matrix in recon_sizes:
                print(run_id, R, N_matrix)
                fname_prj = imdir+f'/R{int(R*1e3):03}mm_recon_prj_{N_matrix}_float32.bin'
                fname_ms = imdir+f'/R{int(R*1e3):03}mm_recon_{N_slices}ms_{N_matrix}_float32.bin'
                recons_prj[run_id][R][N_matrix] = getfile(fname_prj, N_matrix) #.clip(1, None)
                recons_ms[run_id][R][N_matrix] =  getfile(fname_ms, N_matrix) #.clip(1, None)
                
                fname_prj = imdir+f'/R{int(R*1e3):03}mm_paganin_recon_prj_{N_matrix}_float32.bin'
                fname_ms = imdir+f'/R{int(R*1e3):03}mm_paganin_recon_{N_slices}ms_{N_matrix}_float32.bin'
                pagrecons_prj[run_id][R][N_matrix] = getfile(fname_prj, N_matrix) #.clip(1, None)
                pagrecons_ms[run_id][R][N_matrix] =  getfile(fname_ms, N_matrix) #.clip(1, None)
                
    #  read sinograms
    
    beam_width = 7e-3
    channels_2um = np.arange(-beam_width/2, beam_width/2, 2e-6)[:3500] + 1e-6 
    channels_highres = np.arange(-beam_width/2, beam_width/2, 0.5e-6)[:14000] + 0.25e-6
    channels = {'tissuefish_2um': channels_2um, 'tissuefish_highres':channels_highres}

    # #%% read the data
    sinos_prj = {}
    sinos_ms = {}    
    pagsinos_prj = {}
    pagsinos_ms = {}

    for run_id in runs:
        sinos_prj[run_id] = {}
        sinos_ms[run_id] = {}
        pagsinos_prj[run_id] = {}
        pagsinos_ms[run_id] = {}
        
        for R in [0, 0.001, 0.05, 0.1]: #propdists:
            print(run_id, R, 'sino')
            imdir = outdir + run_id 
            
            Ny = 5000
            Nx = {'tissuefish_2um':3500, 'tissuefish_highres':14000}[run_id]
            
            fname_prj = imdir+f'/R{int(R*1e3):03}mm_sino_prj_{Ny}_{Nx}_float32.bin'
            fname_ms = imdir+f'/R{int(R*1e3):03}mm_sino_{N_slices}ms_{Ny}_{Nx}_float32.bin'
            sinos_prj[run_id][R] = getfile(fname_prj, Nx, Ny) 
            sinos_ms[run_id][R] =  getfile(fname_ms,  Nx, Ny) 
            
            if R > 0:
                fname_prj = imdir+f'/R{int(R*1e3):03}mm_paganin_sino_prj_{Ny}_{Nx}_float32.bin'
                fname_ms = imdir+f'/R{int(R*1e3):03}mm_paganin_sino_{N_slices}ms_{Ny}_{Nx}_float32.bin'
                pagsinos_prj[run_id][R] = getfile(fname_prj,  Nx, Ny) 
                pagsinos_ms[run_id][R] =  getfile(fname_ms,  Nx, Ny) 

                
    #%% plot 2x2 - N_matrix, detres
    
    #kw = {'cmap':'bwr'} #, 'vmin':500, 'vmax':1500}
    FSZ = 14
    # simtype = ['prj', '10ms'][0]
    #R = 50e-3
    
    for R in propdists:
        scl = R * 1e2 # scale

        for simtype in ['prj', '10ms']:
            kw = {'1 - prj'  : {'cmap':'bwr', 'vmin':0, 'vmax':1000}, #
                  '1 - 10ms' : {'cmap':'bwr', 'vmin':0, 'vmax':1000},
                  '50 - prj' : {'cmap':'bwr', 'vmin':500, 'vmax':5000},
                  '50 - 10ms': {'cmap':'bwr', 'vmin':500, 'vmax':5000},
                  '100 - prj': {'cmap':'bwr', 'vmin':500, 'vmax':10000},
                  '100 - 10ms':{'cmap':'bwr', 'vmin':500, 'vmax':10000},
            }[f'{int(R*1e3)} - {simtype}']


            fig, ax = plt.subplots(2, 2, figsize=[8, 6])
            if 'vmin' in kw:
                ax[0,0].text(10, 10, f'vmin {kw["vmin"]} / vmax {kw["vmax"]}', color='k', fontweight='bold', verticalalignment='top', horizontalalignment='left')
            for i, run_id in enumerate(runs):
                detres = run_id.split('_')[1]
                ax[0,i].set_title(f'{detres} ({simtype}, {R*1e3} mm)', fontsize=FSZ)
                
                for j, N_matrix in enumerate(recon_sizes):
                    ax[j,0].set_ylabel(N_matrix, fontsize=FSZ)
                    ax[j,i].set_yticks([])
                    ax[j,i].set_xticks([])
                    
                    if simtype == 'prj':
                        img = recons_prj[run_id][R][N_matrix]
                    else:
                        img = recons_ms[run_id][R][N_matrix]
                        
                    m = ax[j,i].imshow(img, **kw)
                    fig.colorbar(m, ax=ax[j,i])
            fig.tight_layout()
            plt.show()
                    
            
#%% plot 1x3 RAW RECON: one detector res + one prop dist, [projection, multislice, difference]
    N_matrix = 4096 
    R = 50e-3

    def WWWL(kw):
        s = f'vmin: {kw["vmin"]} \nvmax: {kw["vmax"]}'
        #WW, WL = int(kw['vmax']-kw['vmin']), int(kw['vmax'] - kw['vmin']/2)
        #s = f'WW: {WW} \nWL: {WL}'
        return s
        
    for run_id in runs:
        
        detres = run_id.split('_')[1]
        #kw = {'tissuefish_highres': {'cmap':'gray', 'vmin':200, 'vmax':4000},   # 2048
        #       'tissuefish_2um':     {'cmap':'gray', 'vmin':200, 'vmax':2000}
        #       }[run_id]
        # kwdiff = {'cmap':'Reds', 'vmax':1000, 'vmin':0}

        kw = {'tissuefish_highres': {'cmap':'gray', 'vmin':300, 'vmax':2500},   # 4096
              'tissuefish_2um':     {'cmap':'gray', 'vmin':200, 'vmax':1300}
              }[run_id]
        kwdiff = {'cmap':'Reds', 'vmax':1000, 'vmin':0}


        fig, ax = plt.subplots(1,3, figsize=[8, 3])
        #ax[0].set_ylabel(f'{detres} - {R*1e3} mm', fontsize=FSZ)
        im_prj = recons_prj[run_id][R][N_matrix]
        im_ms = recons_ms[run_id][R][N_matrix]
        
        ax[0].set_title('Projection approximation', fontsize=FSZ)
        m = ax[0].imshow(im_prj, **kw)
        ax[0].text(Ntxt, Ntxt, WWWL(kw), color='w', fontsize=txtSZ, verticalalignment='top', horizontalalignment='left')
        #fig.colorbar(m, ax=ax[0])

        ax[1].set_title('Multislice', fontsize=FSZ)
        m = ax[1].imshow(im_ms, **kw)
        ax[1].text(Ntxt, Ntxt, WWWL(kw), color='w', fontsize=txtSZ, verticalalignment='top', horizontalalignment='left')
        #fig.colorbar(m, ax=ax[1])

        ax[2].set_title('Absolute difference', fontsize=FSZ)
        diff = absdiff(im_prj, im_ms)
        m = ax[2].imshow(diff, **kwdiff)
        ax[2].text(Ntxt, Ntxt, WWWL(kwdiff), color='k', fontsize=txtSZ, verticalalignment='top', horizontalalignment='left')
        #fig.colorbar(m, ax=ax[2])

        print(f'recon rmse: {np.sqrt(np.mean((im_prj - im_ms)**2)):.3f}')

        for axi in ax:
            axi.set_yticks([])
            axi.set_xticks([])
        fig.tight_layout()
        plt.savefig(figdir + f'recons_{N_matrix}_1x3_R{int(R*1e3):03}mm_{detres}.png')
        plt.show()
                    

            
#%% plot 1x3 PAGANIN: one detector res + one prop dist, [projection, multislice, difference]

    for run_id in runs:
        
        detres = run_id.split('_')[1]
        
        #kw = {'cmap':'bwr', 'vmin':1000, 'vmax':2000}
        # kw = {'tissuefish_highres': {'cmap':'gray', 'vmin':100, 'vmax':2000},
        #       'tissuefish_2um':     {'cmap':'gray', 'vmin':100, 'vmax':2000}
        #       }[run_id]
        # kw = {'tissuefish_highres': {'cmap':'gray', 'vmin':200, 'vmax':4000},
        #       'tissuefish_2um':     {'cmap':'gray', 'vmin':200, 'vmax':2000}
        #       }[run_id]
        kw = {'tissuefish_highres': {'cmap':'gray', 'vmin':250, 'vmax':3000},
              'tissuefish_2um':     {'cmap':'gray', 'vmin':200, 'vmax':1600}
              }[run_id]
        
        fig, ax = plt.subplots(1,3, figsize=[8, 3])
        #ax[0].set_ylabel(f'{detres} - {R*1e3} mm', fontsize=FSZ)

        im_prj = pagrecons_prj[run_id][R][N_matrix]
        im_ms = pagrecons_ms[run_id][R][N_matrix]
        
        ax[0].set_title('Projection approximation', fontsize=FSZ)
        m = ax[0].imshow(im_prj, **kw)
        ax[0].text(Ntxt, Ntxt, WWWL(kw), color='w', fontsize=txtSZ, verticalalignment='top', horizontalalignment='left')
        #fig.colorbar(m, ax=ax[0])

        ax[1].set_title('Multislice', fontsize=FSZ)
        m = ax[1].imshow(im_ms, **kw)
        ax[1].text(Ntxt, Ntxt, WWWL(kw), color='w', fontsize=txtSZ, verticalalignment='top', horizontalalignment='left')
        #fig.colorbar(m, ax=ax[1])

        ax[2].set_title('Absolute difference', fontsize=FSZ)
        kwdiff = {'cmap':'Reds', 'vmax':1000, 'vmin':0}
        diff = absdiff(im_prj, im_ms)
        m = ax[2].imshow(diff, **kwdiff)
        ax[2].text(Ntxt, Ntxt, WWWL(kwdiff), color='k', fontsize=txtSZ, verticalalignment='top', horizontalalignment='left')
        #fig.colorbar(m, ax=ax[2])
        
        print(f'paganin rmse: {np.sqrt(np.mean((im_prj - im_ms)**2)):.3f}')

        for axi in ax:
            axi.set_yticks([])
            axi.set_xticks([])
        fig.tight_layout()
        plt.savefig(figdir + f'recons_{N_matrix}_paganin_1x3_R{int(R*1e3):03}mm_{detres}.png')
        plt.show()
        

        
#%% plot RAW SINOGRAMS

    for run_id in runs:
        R = 50e-3
        print(run_id)
    
        sino_prj = sinos_prj[run_id][R]
        sino_ms = sinos_ms[run_id][R]
    
        kw = {'tissuefish_highres': {'cmap':'gray', 'aspect':'auto', 'vmin':0, 'vmax':0.3},
              'tissuefish_2um':     {'cmap':'gray', 'aspect':'auto', 'vmin':0, 'vmax':0.3}
              }[run_id]
        
        fig, ax = plt.subplots(1,3, figsize=[7, 3])
        #ax[0].set_ylabel(f'{detres} - {R*1e3} mm', fontsize=FSZ)
    
        
        ax[0].set_title('Projection approximation', fontsize=FSZ)
        m = ax[0].imshow(sino_prj, **kw)
        #ax[0].text(Ntxt, Ntxt, WWWL(kw), color='w', fontsize=txtSZ, verticalalignment='top', horizontalalignment='left')
        fig.colorbar(m, ax=ax[0])
    
        ax[1].set_title('Multislice', fontsize=FSZ)
        m = ax[1].imshow(sino_ms, **kw)
        #ax[1].text(Ntxt, Ntxt, WWWL(kw), color='w', fontsize=txtSZ, verticalalignment='top', horizontalalignment='left')
        fig.colorbar(m, ax=ax[1])
    
        ax[2].set_title('Difference', fontsize=FSZ)
        #kwdiff = {'cmap':'Reds', 'aspect':'auto', 'vmin':0, 'vmax':.1}
        kwdiff = {'cmap':'bwr', 'aspect':'auto', 'vmin':-.015, 'vmax':.015}
        diff = sino_prj- sino_ms#absdiff(sino_prj, sino_ms)
        m = ax[2].imshow(diff, **kwdiff)
        #ax[2].text(Ntxt, Ntxt, WWWL(kwdiff), color='k', fontsize=txtSZ, verticalalignment='top', horizontalalignment='left')
        fig.colorbar(m, ax=ax[2])
        
    
        for axi in ax:
            axi.set_yticks([])
            axi.set_xticks([])
        fig.tight_layout()
        #plt.savefig(figdir + f'sinos_R{int(R*1e3):03}mm_{detres}.png')
        plt.show()



#%% plot SINO LINE PROFILES

    #i_view = 2000

    for i_view in [0, 1000, 2000, 3000, 4000]:
    #for run_id in [runs[1]]: #runs:
        R = 50e-3
        detchannels = channels[run_id]
        print(run_id)
    
        sino_prj = sinos_prj[run_id][R]
        sino_ms = sinos_ms[run_id][R]
    
        kw = {'tissuefish_highres': {'cmap':'gray', 'aspect':'auto', 'vmin':0, 'vmax':0.3},
              'tissuefish_2um':     {'cmap':'gray', 'aspect':'auto', 'vmin':0, 'vmax':0.3}
              }[run_id]
        
        fig, ax = plt.subplots(1,2, figsize=[6, 3])
        #ax[0].set_ylabel(f'{detres} - {R*1e3} mm', fontsize=FSZ)
    
        # ax[0].set_title('Sino difference', fontsize=FSZ)
        # kwdiff = {'cmap':'bwr', 'aspect':'auto', 'vmin':-.015, 'vmax':.015}
        # diff = sino_prj- sino_ms
        # m = ax[0].imshow(diff, **kwdiff)
        # fig.colorbar(m, ax=ax[0])
        # #ax[0].set_yticks([])
        # ax[0].set_xticks([])
            
        ax[1].set_title(f'View {i_view+1}')
        ax[1].plot(detchannels, sino_prj[i_view]-sino_ms[i_view], 'k-', lw=.1, label='difference')
        #ax[1].plot(detchannels, sino_prj[i_view], 'k-', lw=.2, label='projection')
        #ax[1].plot(detchannels, sino_ms[i_view], 'r-', lw=.1, label='multislice')
        ax[1].legend()
        
        fig.tight_layout()
        plt.show()



#%% plot differences in the sinograms

#for R in propdists: 
    R = 0
    fig, ax = plt.subplots(1,2, figsize=[4, 3], sharey=True)
    ax[0].set_ylabel('view angle [deg]')

    for i, run_id in enumerate(runs):
        ax[i].set_xlabel('detector channel [mm]')
    
        sino_prj = sinos_prj[run_id][R]
        sino_ms = sinos_ms[run_id][R]
    
        detchannels = channels[run_id] * 1e3
        kw = {'cmap':'gray', 'aspect':'auto', 'vmin':0, 'vmax':0.3,
              'extent':(detchannels[0], detchannels[-1], 0, 180)}
        #kwdiff = {'cmap':'bwr', 'aspect':'auto', 'vmin':-.02, 'vmax':.02,
        kwdiff = {'cmap':'bwr', 'aspect':'auto', 'vmin':-.005, 'vmax':.005,  # EXIT WAVE
                  'extent':(detchannels[0], detchannels[-1], 0, 180)}
        title = {'tissuefish_2um': '2-$\mu$m', 'tissuefish_highres':'0.5-$\mu$m'}[run_id]
        ax[i].set_title(title, fontsize=FSZ)

        diff = sino_prj - sino_ms
        m = ax[i].imshow(diff, **kwdiff)
    cax = fig.add_axes([1.0, 0.152, 0.027, 0.735])
    cbar = fig.colorbar(m, cax=cax, ticks=np.arange(kwdiff['vmin'], kwdiff['vmax']+1e-8, kwdiff['vmax']))
    

    fig.tight_layout()
    plt.savefig(figdir + f'sinodiffs_R{int(R*1e3):03}mm.png', bbox_inches='tight')
    plt.show()



#%% sino diff - stacked 

#for R in propdists:

    R = 0
    fig, ax = plt.subplots(2, 1, figsize=[3.5, 5.5])

    for i, run_id in enumerate(runs):
        ax[i].set_xlabel('channel [mm]')
        ax[i].set_ylabel('view angle [deg]')

        sino_prj = sinos_prj[run_id][R]
        sino_ms = sinos_ms[run_id][R]
    
        detchannels = channels[run_id] * 1e3
        kw = {'cmap':'gray', 'aspect':'auto', 'vmin':0, 'vmax':0.3,
              'extent':(detchannels[0], detchannels[-1], 0, 180)}
        #kwdiff = {'cmap':'bwr', 'aspect':'auto', 'vmin':-.02, 'vmax':.02,
        kwdiff = {'cmap':'bwr', 'aspect':'auto', 'vmin':-.005, 'vmax':.005,  # EXIT WAVE
                  'extent':(detchannels[0], detchannels[-1], 0, 180)}
        title = {'tissuefish_2um': '2-$\mu$m', 'tissuefish_highres':'0.5-$\mu$m'}[run_id]
        ax[i].set_title(title+' detector', fontsize=FSZ)

        diff = sino_prj - sino_ms
        m = ax[i].imshow(diff, **kwdiff)
        fig.colorbar(m, ax=ax[i])

    fig.tight_layout()
    plt.savefig(figdir + f'sinodiffs_stacked_R{int(R*1e3):03}mm.png', bbox_inches='tight')
    plt.show()










    #     #%%  no propagation or phase retrieval

        
    #     fig,ax=plt.subplots(1, 1, figsize=[4,3])
    #     #m = ax.imshow(recons_prj['tissuefish_2um'][1e-3][4096], cmap='gray', vmin=0, vmax=100)
    #     m = ax.imshow(recons_prj['tissuefish_highres'][1e-3][4096], cmap='gray', vmin=0, vmax=300)
    #     fig.colorbar(m)
    #     plt.show()
    # #%%
        
    #     fig,ax=plt.subplots(1, 1, figsize=[4,3])
    #     m = ax.imshow(pagrecons_prj['tissuefish_2um'][1e-3][4096], cmap='gray', vmin=0, vmax=100)
    #     fig.colorbar(m)
    #     plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        