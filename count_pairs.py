import frogress
import pandas as pd
import numpy as np
import sys
from subprocess import call
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
import glob
from astropy.io import fits
import healpy as hp


# ra,dec 2 xyz
def get_xyz(ra,dec):
    theta = (90-dec)*np.pi/180
    phi = ra*np.pi/180
    z = np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    x = np.cos(phi)*np.sin(theta)
    return x,y,z


# getting mirror points
@np.vectorize
def get_mirr(ra0,dec0,gra,gdec):
    x0,y0,z0  = get_xyz(ra0,dec0)
    xg,yg,zg  = get_xyz(gra,gdec)
    cos_theta = x0*xg + y0*yg + z0*zg 
    alpha   = y0*zg - yg*z0  
    beta    = x0*zg - xg*z0  
    gamma   = yg*x0 - xg*y0  
    
    num_zm = (2*cos_theta**2 - 1)*(x0*beta + y0*alpha) - cos_theta*(xg*beta + yg*alpha)
    den_zm = yg*(gamma*x0 - alpha*z0) - xg*(beta*z0 + gamma*y0) + zg*(beta*x0 + alpha*y0)
    
    zm = num_zm *1.0/den_zm
    ym = (alpha*cos_theta - zm*(z0*alpha - x0*gamma))*1.0/(x0*beta + y0*alpha)
    xm = (beta*cos_theta - zm*(y0*gamma + z0*beta))*1.0/(x0*beta + y0*alpha)
    r = (xm**2 + ym**2 + zm**2)**0.5
        
    theta = np.arccos(np.clip(zm*1.0/r,-1,1))
    phi = np.arctan(ym*1.0/xm)
    if xm>0 and ym>0:
        phi = phi
    if xm<0 and ym>0:
        phi = np.pi - abs(phi)
    if xm<0 and ym<0:
        phi = np.pi + abs(phi)
    if xm>0 and ym<0:
        phi = 2*np.pi - abs(phi)
        
    ra = phi*180/np.pi
    dec = (np.pi/2 - theta)*180.0/np.pi
    
    return ra,dec




# read lens file
def lens_select(Rmin, Rmax, Njacks=30, mirror=False):
    ppath = '/mnt/home/faculty/csurhud/github/weaklens_pipeline/DataStore/redmapper'
    cdat = fits.open('%s/redmapper_dr8_public_v6.3_catalog.fits'%ppath)[1].data
    mdat = fits.open('%s/redmapper_dr8_public_v6.3_members.fits'%ppath)[1].data

    idx = (cdat['p_cen'][:,0]>0.96) & (cdat['Z_LAMBDA_ERR']*1.0/(1.0+cdat['Z_LAMBDA']) < 0.01)
    pid = cdat['id'][idx]
    pra = cdat['ra_cen'][idx,0]
    pdec = cdat['dec_cen'][idx,0]
    zspec = cdat['z_spec'][idx]
    zlamb = cdat['Z_LAMBDA'][idx]
    pzred = 0.0*pra
    idx = (zspec>0)
    pzred[np.arange(len(pzred))[idx]] = zspec[idx] 
    pzred[np.arange(len(pzred))[~idx]] = zlamb[~idx]

    #........putting the jackknife on the parents............#
    np.random.seed(123)
    pjkreg = np.random.randint(Njacks, size=len(pra))
    #.......................................................#

    flg = np.isin(mdat['id'], pid)     
    mra   = mdat['ra']  [flg]
    mdec  = mdat['dec'] [flg]
    mrsep = mdat['r']   [flg]
    mpid  = mdat['id']  [flg]

    mzred = 0.0*mra    
    midx  = np.isin(pid, mpid)
    mzred = pzred[np.arange(len(midx))[midx]] 
    mrsep = mrsep*(1.0 + mzred)
    mpra  = pra[midx]
    mpdec = pdec[midx]
    mjkreg = pjkreg[midx]

    idx = (mrsep>Rmin) & (mrsep<Rmax) 

    mra     = mra[idx]
    mdec    = mdec[idx]
    mzred   = mzred[idx]
    mpra    = mpra[idx]
    mpdec   = mpdec[idx]
    mjkreg  = mjkreg[idx]

    mirrra, mirrdec = get_mirr(mpra, mpdec, mra, mdec)

    msk = hp.read_map("/mnt/home/student/cdivya/github/weaklens_pipeline_s16a/DataStore/data/S16A_mask_w_holes.fits")
    galpix = hp.ang2pix(int(np.sqrt(msk.size/12)), mra, mdec, lonlat=1)
    mirrpix = hp.ang2pix(int(np.sqrt(msk.size/12)), mirrra, mirrdec, lonlat=1)
    
    sel = (msk[galpix]==1.0) & (msk[mirrpix]==1.0)
    
    if mirror==False:
        ra    = mra[sel].astype('float')
        dec   = mdec[sel].astype('float')
    else:
        ra    = mirrra[sel].astype('float')
        dec   = mirrdec[sel].astype('float')
        print "using mirror points"


    zred  = mzred[sel].astype('float')
    mjkreg = mjkreg[sel]
    wt    = ra*1.0/ra
    sys.stdout.write("Selecting %d samples \n" % (ra.size))
    return ra, dec, zred, mjkreg, wt
 


# read sources
def read_sources(ifil, use_srcrand=False):
    # various columns in sources 
    # ragal, decgal
    if use_srcrand:
        data = pd.read_csv(ifil, delim_whitespace=1)
    else:
        data = fits.open(ifil)[1].data

    sra = data['ira']
    sdec = data['idec']
    mat = np.transpose([sra,sdec])
    return mat



# counts given radial bins and lens positions
def run_pipe(Omegam=0.315, rmin=0.2, rmax=2.0, nbins=10, outputfile = 'pairs.dat', Rmin=0.3, Rmax=0.6, Njacks=30, mirror=False, use_srcrand=False):
    #set the cosmology with omegaM parameter
    cc = FlatLambdaCDM(H0=100, Om0=Omegam) # fixing H0=100 to set units in Mpc h-1
    # set the projected radial binning
    rmin  =  rmin
    rmax  =  rmax
    nbins = nbins #10 radial bins for our case
    rbins  = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    rdiff  = np.log10(rbins[1]*1.0/rbins[0])
    # initializing arrays for signal compuations
    sumwls        = np.zeros((len(rbins[:-1]), Njacks))
    #sumwls        = np.zeros(len(rbins[:-1]))

    # getting the lenses data
    lra, ldec, lred, ljkreg, lwgt = lens_select(Rmin=Rmin, Rmax=Rmax, Njacks=30, mirror=mirror)

    # convert lense ra and dec into x,y,z cartesian coordinates
    lx, ly, lz = get_xyz(lra, ldec)

    # putting kd tree around the lenses
    lens_tree = cKDTree(np.array([lx, ly, lz]).T)

    print('lenses tree is ready\n')

    # setting maximum search radius
    dcommin = cc.comoving_distance(np.min(lred)).value
    dismax  = (rmax*1.0/(dcommin))

    # lets first catch the file list for source
    if use_srcrand:
        print "using source randoms"
        from glob import glob
        sflist = np.sort(glob('/mnt/home/student/cdivya/github/redmapperxhsc16a/hsc_randoms/hsc_randoms/rand_*.dat'))
    else:
        sflist = ['AEGIS', 'VVDS', 'WIDE12H', 'XMM', 'GAMA15H', 'GAMA09H', 'HECTOMAP']


    # Ready to pounce on the source data
    for ifil in sflist:
        if use_srcrand:
            fpin = ifil
        else:
            fpin ='/mnt/home/faculty/csurhud/github/weaklens_pipeline/DataStore/S16A_v2.0/%s.fits'%ifil

        # catching the source data matrix
        # please have a check for the columns names

        # please have a check for the columns names
        datagal = read_sources(fpin, use_srcrand=use_srcrand)
        Ngal = len(datagal[:,0])  # total number of galaxies in the source file
        # first two entries are ra and dec for the sources
        allragal  = datagal[:,0]
        alldecgal = datagal[:,1]
        # ra and dec to x,y,z for sources
        allsx, allsy, allsz = get_xyz(allragal, alldecgal)
        # query in a ball around individual sources and collect the lenses ids with a maximum radius
        slidx = lens_tree.query_ball_point(np.transpose([allsx, allsy, allsz]), dismax)
        # various columns in sources
        # ragal, decgal
        # looping over all the galaxies
        for igal in range(Ngal):
            ragal    = datagal[igal,0]
            decgal   = datagal[igal,1]
            wgal     = ragal*1.0/ragal

            # array of lenses indices
            lidx = np.array(slidx[igal])
            # removing sources which doesn't have any lenses around them
            if len(lidx)==0:
                continue
            sra    = ragal
            sdec   = decgal

            l_ra    = lra[lidx]
            l_dec   = ldec[lidx]
            l_zred  = lred[lidx]
            l_jkreg = ljkreg[lidx]
            l_wgt   = lwgt[lidx]

            sx, sy, sz = get_xyz(sra,sdec) # individual galaxy ra,dec-->x,y,z
            lx, ly, lz = get_xyz(l_ra,l_dec) # individual galaxy ra,dec-->x,y,z

            # getting the radial separations for a lense source pair
            sl_sep = np.sqrt((lx - sx)**2 + (ly - sy)**2 + (lz - sz)**2)
            sl_sep = sl_sep * cc.comoving_distance(l_zred).value
            for ll,sep in enumerate(sl_sep):
                if sep<rmin or sep>rmax:
                    continue
                rb = int(np.log10(sep*1.0/rmin)*1/rdiff)
                # to get the area just have to multiply paircounts by dcom**2 in randomsource case
 
                # following equations given in the surhud's lectures
                if use_srcrand==True:
                    w_ls    = cc.comoving_distance(l_zred[ll]).value**2 * l_wgt[ll] * wgal 
                else:
                    w_ls    = l_wgt[ll] * wgal 
                # separate numerator and denominator computation
                sumwls[rb][l_jkreg[ll]]  += w_ls

        print(ifil)

    
    if mirror:
        outputfile = outputfile + '_mirror'
    fout = open(outputfile, "w")
    fout.write("# 0:rmin/2+rmax/2 1:paircounts 2: jackknife\n")
    for i in range(len(rbins[:-1])):
        rrmin = rbins[i]
        rrmax = rbins[i+1]

        for jkreg in range(Njacks):
            idx = (np.arange(Njacks) != jkreg)
            #Resp = sumwls_resp[i]*1.0/sumwls[i]
            fout.write("%le\t%le\t%le\n"%(rrmin/2.0+rrmax/2.0, sum(sumwls[i][idx]), jkreg))
    fout.write("#OK")
    fout.close()

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", help="output file", type=str, default='pairs.dat')
    parser.add_argument("--mirror", help="use mirror points", type=bool, default=False)
    parser.add_argument("--srcrand", help="use source randoms", type=bool, default=False)

    args = parser.parse_args()

    run_pipe(rmin=0.02, rmax=1.0, nbins=10, Rmin=0.3, Rmax=0.6, mirror = args.mirror, outputfile = args.output, use_srcrand = args.srcrand)


