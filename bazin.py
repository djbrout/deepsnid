import numpy as np
np.set_printoptions(threshold=np.nan)
from scipy.optimize import curve_fit
from lmfit import Model
import sys
import matplotlib.pyplot as plt

def bazinfunc(t, t0, tau_fall, tau_rise, A, B):
    #http://arxiv.org/pdf/0904.1066v1.pdf Equation #1
    return A*(np.exp(-1.*(t-t0)/tau_fall)/(1.+np.exp((t-t0)/tau_rise))) + B

def fit(mjd,flux,fluxerr):
    t0_guess = mjd[np.argmax(flux)]
    tau_fall_guess = 10.
    tau_rise_guess = 10.
    A = 1.
    B = 1.
    p0 = [t0_guess, tau_fall_guess, tau_rise_guess, A, B]
    popt, pcov, infodict, errmsg, ier = curve_fit(bazinfunc, mjd, flux,
                                                  sigma=fluxerr, p0=p0,maxfev=2000000,full_output=True)
    chisq = (infodict['fvec'] ** 2).sum() / (len(infodict['fvec'])-len(popt))
    #fmodel = bazinfunc(flux, *popt)
    #chisq = chisquare(flux,fmodel,sd=fluxerr)

    return chisq,popt

def lmfit(mjd,flux,fluxerr):
    t0_guess = mjd[np.argmax(flux)]
    tau_fall_guess = 40.
    tau_rise_guess = -5.
    A = 150.
    B = 20.
    # nflux = np.zeros(2+len(np.array(flux)))
    # nfluxerr = np.ones(2+len(np.array(flux)))/10.
    # nmjd = np.zeros(2+len(np.array(flux)))
    #
    # nflux[1:-1] = flux
    # nfluxerr[1:-1] = fluxerr
    # nmjd[1:-1] = mjd
    # nmjd[1] = mjd[0]-100.
    # nmjd[-1] = mjd[-1]+150
    #
    # flux = nflux
    # fluxerr = nfluxerr
    # mjd = nmjd

    bmod = Model(bazinfunc)
    bmod.set_param_hint('t0', value=t0_guess, min=t0_guess-20, max=t0_guess+20)
    bmod.set_param_hint('tau_fall', value=tau_fall_guess)
    bmod.set_param_hint('tau_rise', value=tau_rise_guess)
    bmod.set_param_hint('A',value=A)
    bmod.set_param_hint('B',value=B)

    pars = bmod.make_params()
    #print(bmod.param_names)
    #print(bmod.independent_vars)
    # print(np.array(flux))
    # print(np.array(1./np.array(fluxerr)))
    # print(np.array(mjd))
    result = bmod.fit(np.array(flux),method='leastsq',weights=1./np.array(fluxerr), t=np.array(mjd))

    #print(result.fit_report())
    # plt.clf()
    # plt.errorbar(np.array(mjd), np.array(flux), yerr=fluxerr,fmt='o')
    # plt.plot(np.array(mjd), result.init_fit, 'k--')
    # plt.plot(np.array(mjd), result.best_fit, 'r-')
    # #plt.xlim(mjd[1],mjd[-2])
    # plt.savefig('bazinfit.png')



    chisq = result.redchi
    ps = result.best_values
    popt = [ps['t0'],ps['tau_fall'],ps['tau_rise'],ps['A'],ps['B']]
    #print('popt',popt)
    #sys.exit()
    # if chisq < 2.:
    #     input('good chisq!')

    # popt, pcov, infodict, errmsg, ier = curve_fit(bazinfunc, mjd, flux,
    #                                               sigma=fluxerr, p0=p0, maxfev=2000000, full_output=True)
    #
    # chisq = (infodict['fvec'] ** 2).sum() / (len(infodict['fvec']) - len(popt))
    return chisq,popt

def chisquare(ydata, ymod, sd=None):

    if sd is None:
        redchisq = np.nansum((ydata - ymod) ** 2)/len(ydata)
    else:
        redchisq = np.nansum(((ydata - ymod) / sd) ** 2)/len(ydata)

    return redchisq

if __name__ == '__main__':
    pass