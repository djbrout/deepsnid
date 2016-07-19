import os
import matplotlib as m

m.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.nan)
import sys
import dilltools as dt
import sncosmo
import bazin


#SALT AND BAZIN MUST BE THE FIRST TWO SOURCES AND ONLY IF THEY WORK WILL THE REST OF THE SOURCES BE FIT!
sources = [    'salt2', 's11-2004hx', 's11-2005lc', 's11-2005hl', 's11-2005hm',
               's11-2005gi', 's11-2006fo', 's11-2006jl', 'hsiao', 'hsiao-subsampled', 'snf-2011fe', 'snf-2011fe',
               'snana-2004gq', 'snana-sdss004012', 'snana-2006fo', 'snana-sdss014475', 'snana-2006lc', 'snana-2007ms',
               'snana-04d1la', 'snana-04d4jv', 'snana-2004gv', 'snana-2006ep', 'snana-2007y', 'snana-2004ib',
               'snana-2005hm', 'snana-2006jo', 'snana-2007nc', 'snana-2004hx', 'snana-2005gi', 'snana-2006gq',
               'snana-2006kn', 'snana-2006jl', 'snana-2006iw', 'snana-2006kv', 'snana-2006ns', 'snana-2007iz',
               'snana-2007nr', 'snana-2007kw', 'snana-2007ky', 'snana-2007lj', 'snana-2007lb', 'snana-2007ll',
               'snana-2007nw', 'snana-2007ld', 'snana-2007md', 'snana-2007lz', 'snana-2007lx', 'snana-2007og',
               'snana-2007ny', 'snana-2007nv', 'snana-2007pg', 'snana-2006ez', 'snana-2006ix', 'whalen-z15b',
               'whalen-z15g', 'whalen-z25b', 'whalen-z25d', 'whalen-z25g', 'whalen-z40b', 'whalen-z40g',
               'nugent-hyper','nugent-sn2p','nugent-sn2l','nugent-sn2n','mlcs2k2'
               ]


def grabdata(loc,list):
    f = open(list,'r').readlines()
    files = []
    for file in f:
        files.append(os.path.join(loc,file.strip()))

    lightcurves = []
    hostzs = []
    hostz_stds = []
    snids = []
    sntypes = []
    params = []

    max = 20000
    cntr = 0

    for file in files:
        cntr += 1
        if cntr > max:
            continue
        sn = open(file,'r').readlines()
        keepgoing = True
        i = 0
        sntype = -999
        snid = -999
        hostz = -999
        hostz_std = -999
        while keepgoing:
            if 'SNID:' in sn[i]:
                snid = sn[i].split()[1]
            if 'SNTYPE:' in sn[i]:
                sntype = sn[i].split()[1]
            if 'HOST_GALAXY_PHOTO-Z:' in sn[i]:
                hostz = float(sn[i].split()[1])
                hostz_std = float(sn[i].split()[3])
                keepgoing = False
            i += 1

        hostzs.append(hostz)
        hostz_stds.append(hostz_std)
        snids.append(snid)
        sntypes.append(sntype)

        sndata = dt.read(file,23,24)
        flux = sndata['FLUXCAL']
        fluxerr = sndata['FLUXCALERR']
        filt = sndata['FLT']
        mjd = sndata['MJD']

        badrow = -9
        for i,f,fe,flt,m in zip(range(len(flux)),flux,fluxerr,filt,mjd):
            if 'g+r+i+z' in str(f):
                badrow = i
            elif 'g+r+i+z' in str(fe):
                badrow = i
            elif 'g+r+i+z' in str(flt):
                badrow = i
            elif 'g+r+i+z' in str(m):
                badrow = i
        if badrow != -9:
            flux = np.delete(flux,badrow)
            fluxerr = np.delete(fluxerr,badrow)
            filt = np.delete(filt, badrow)
            mjd = np.delete(mjd, badrow)

        cosmofile = file.split('.')[0]+'.sncosmo'
        out = open(cosmofile,'w')
        out.write('time band flux fluxerr zp zpsys\n')
        for i,f,fe,flt,m in zip(range(len(flux)),flux,fluxerr,filt,mjd):
            out.write(str(m)+' des'+str(flt)+' '+str(f)+' '+str(fe)+' 27.5 ab\n')
        out.close()
        lc = sncosmo.read_lc(cosmofile)
        lightcurves.append(lc)

        params.append({'snid':snid,'type':sntype,'hostz':hostz,'hostz_std':hostz_std})
        print(snid,sntype,hostz,hostz_std)

    hostzs = np.array(hostzs)
    hostz_stds = np.array(hostz_std)
    snids = np.array(snids)
    sntypes = np.array(sntypes,dtype='float')

    print('unique types',np.unique(sntypes))

    dsntypes = np.zeros(len(sntypes))
    dsntypes[sntypes > 1] = np.ones(len(sntypes[sntypes > 1]))
    dsntypes[sntypes == 1] = np.zeros(len(sntypes[sntypes == 1]))
    dsntypes[sntypes == -9] = np.zeros(len(sntypes[sntypes == -9]))-9

    print('nonia',len(dsntypes[dsntypes == 1]))
    print('ia',len(dsntypes[dsntypes == 0]))
    print('none',len(dsntypes[dsntypes == -9]))

    'FEED ONLY LCs WITH NON -9 TYPES THROUGH ALL THE LIGHCURVE FITS AND SPLIT THAT INTO TEST AND TRAIN'
    knownlightcurves = []
    knownparams = []
    knowntypes = []
    blindlightcurves = []
    blindparams = []
    for lc,t,p in zip(lightcurves,dsntypes,params):
        if t == -9:
            blindlightcurves.append(lc)
            blindparams.append(p)
        else:
            knownlightcurves.append(lc)
            knownparams.append(p)
            knowntypes.append(t)

    ndim = 1 + 1 + 1 + len(sources) + 2
    fitparams = np.zeros((len(knownlightcurves), ndim)) #THIS NEEDS TO CUT OUT -9S

    fitparams, truetypes = lcfit(knownlightcurves, knownparams, fitparams, knowntypes, sources)
    print('fitparams.shape', fitparams.shape)
    print('truetypes.shape', truetypes.shape)
    np.savez('sncc_known2.npz', big_data_array=fitparams, truetypes=truetypes, params=knownparams)
    print('saved sncc_known2.npz')
    return



def lcfit(lightcurves,params,fitparams,truearray,sources):
    models = []
    for source in sources:
        try:
            models.append(sncosmo.Model(source=source))
        except:
            models.append('custom_bazin_model')
    iatot = 0.
    noniatot = 0.
    iafailed = 0.
    noniafailed= 0.
    badrows = []
    for i,lc in enumerate(lightcurves):
        if truearray[i] == 0:
            iatot += 1.
        else:
            noniatot += 1.

        z = params[i]['hostz']
        ze = params[i]['hostz_std']
        fitparams[i,-2] = z
        fitparams[i,-1] = ze
        t0 = lc['time'][np.argmax(lc['flux'])]
        print('Fitting LC '+str(i)+'/'+str(len(lightcurves)),'t0',t0,'z',z)
        input_param_index = 3

        saltworked = True
        bazinworked = True
        for j, model, source in zip(range(len(sources)), models, sources):

            if source == 'salt2':
                try:
                    #model.set(z=z)
                    res, fitted_model = sncosmo.fit_lc(lc, model,['z','t0', 'x0', 'x1', 'c'],
                                               bounds={'z':(z-3.*ze,z+3.*ze ),'t0':(t0-50.,t0+50.)})
                    #print(res.keys())
                    print('Type',truearray[i],'Source', source,'Fit Chisq', res.chisq / res.ndof,
                          'Fit c',res.parameters[4],'Fit x0',res.parameters[2],'Fit x1',res.parameters[3])
                    #print('chisq/ndof',res.chisq/res.ndof)
                    #print('param names',res.param_names[:])
                    #print('params',res.parameters[:])
                    #print('-'*100)

                    fitparams[i,:4] = np.array([res.parameters[2], res.parameters[3], res.parameters[4], res.chisq/res.ndof])
                    t0 = res.parameters[1]
                    #fitparams[i, :] = np.array([res.parameters[4], res.chisq / res.ndof])
                    #sys.exit()
                except sncosmo.fitting.DataQualityError:
                    #fitparams[i, :] = np.array([99, 99, 99, 99])
                    # if truearray[i] == 0:
                    #     print('could not fit Type Ia')
                    #     sys.exit()
                    print('fitting failed: SN Type ',truearray[i])
                    fitparams[i, :4] = np.array([0., 0., 0., 500.])
                    print('-'*100)
                    if truearray[i] == 0:
                        iafailed += 1.
                    else:
                        noniafailed += 1.
                    #badrows.append(i)
                    #saltworked = False

                except RuntimeError:
                    # if truearray[i] == 0:
                    #     print('could not fit Type Ia')
                    #     sys.exit()
                    #fitparams[i, :] = np.array([99, 99, 99, 99])
                    fitparams[i, :4] = np.array([0., 0., 0., 500.])
                    print('fitting failed: SN Type ',truearray[i])
                    print('-'*100)
                    #badrows.append(i)
                    #saltworked = False
                    if truearray[i] == 0:
                        iafailed += 1.
                    else:
                        noniafailed += 1.
            elif saltworked:
                if source == 'bazin':
                    if True:
                        flux = lc['flux']
                        fluxerr = lc['fluxerr']
                        bands = lc['band']
                        mjds = lc['time']
                        input_param_index = 3
                        for band in np.sort(np.unique(bands)):
                            bflux = flux[bands==band]
                            bfluxerr = fluxerr[bands==band]
                            bmjds = mjds[bands==band]
                            try:
                                chisq,popt = bazin.lmfit(bmjds,bflux,bfluxerr)
                                if chisq > 500.:
                                    chisq = 500.
                                print('Source Bazin',band,'Fit Chisq',chisq,'t0',popt[0],'tau_fall',popt[1],
                                      'tau_rise',popt[2],'A',popt[3])
                            except:
                                print('BASIN FAILED ' * 10)
                                chisq, popt = 500.,[0,0,0,0,0]
                            fitparams[i,j+input_param_index] = chisq
                            input_param_index += 1
                            fitparams[i, j+input_param_index] = popt[1]
                            input_param_index += 1
                            fitparams[i, j+input_param_index] = popt[2]
                            input_param_index += 1
                            fitparams[i, j+input_param_index] = popt[3]
                            input_param_index += 1
                            fitparams[i, j+input_param_index] = popt[4]
                            input_param_index += 1
                        input_param_index -= 1

                else:
                    if bazinworked:
                        if not source == 'mlcs2k2':
                            #print(model.param_names)
                            try:
                                res, fitted_model = sncosmo.fit_lc(lc, model, ['z', 't0', 'amplitude'],
                                                                   bounds={'z':(z-3.*ze,z+3.*ze ),
                                                                           't0': (t0 - 20., t0 + 20.)})
                                                                   #bounds={'z': (z - .05 * z,
                                                                   #              z + .05 * z)})  # assume we know hostgal z to within 5%
                                print('Source',source,'Fit Chisq',res.chisq/res.ndof)
                                fitparams[i, j+input_param_index] = res.chisq/res.ndof
                            except:
                                print('Source', source, 'Could not Fit... Chisq', 300.)
                                fitparams[i, j+input_param_index] = 300.
                        else:
                            try:
                                res, fitted_model = sncosmo.fit_lc(lc, model, ['z', 't0', 'amplitude', 'delta',],
                                                                   bounds={'z':(z-3.*ze,z+3.*ze ),
                                                                           't0': (t0 - 20., t0 + 20.)})
                                                                   #bounds={'z': (z - .05 * z,
                                                                   #              z + .05 * z)})  # assume we know hostgal z to within 5%
                                print('Source', source, 'Fit Chisq', res.chisq / res.ndof)
                                fitparams[i, j + input_param_index] = res.chisq / res.ndof
                            except:
                                print('Source', source, 'Could not Fit... Chisq', 300.)
                                fitparams[i, j + input_param_index] = 300.

        print(fitparams[i,:])
        print('-' * 100)

    print('Percentage of Ia Failed',iafailed/iatot)
    print('Percentage of NONIa Failed',noniafailed/noniatot)
    fitparams = np.delete(fitparams,np.array(badrows),axis=0)
    truetypes = np.delete(truearray,np.array(badrows))


    return fitparams, truetypes




if __name__ == '__main__':

    hostloc = '/Volumes/ExtraSpace/DES_BLIND+HOSTZ'
    hostlist = '/Volumes/ExtraSpace/DES_BLIND+HOSTZ/DES_BLIND+HOSTZ.LIST'

    nohostloc = '/Volumes/ExtraSpace/DES_BLINDnoHOSTZ'
    nohostlist = '/Volumes/ExtraSpace/DES_BLINDnoHOSTZ/DES_BLINDnoHOSTZ.LIST'

    grabdata(hostloc,hostlist)