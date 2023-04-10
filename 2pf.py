import numpy as np
import matplotlib.pyplot as plt

rad, dd, jkid = np.loadtxt('pairs.dat', unpack=1)
rad, md, jkid = np.loadtxt('pairs.dat_mirror', unpack=1)
rad, dr, jkid = np.loadtxt('random_pairs.dat', unpack=1)
rad, mr, jkid = np.loadtxt('random_pairs.dat_mirror', unpack=1)


njacks  = len(np.unique(jkid))
print njacks
dens        = dd*1/(dr/100.0 * (np.pi/(60*180))**2) - md*1/(mr/100.0 * (np.pi/(60*180))**2)
#dens    = (dd - md)*1.0/md
dens        = dens.reshape((-1,njacks))


print np.mean(rad.reshape((-1,njacks)), axis=1)

import sys
xx      = np.unique(rad)
yy      = np.mean(dens, axis=1)

cov     = np.zeros((len(xx),len(xx)))
for ii in range(len(xx)):
    for jj in range(len(xx)):
        cov[ii][jj] = np.mean((dens[ii,:] - yy[ii])*(dens[jj,:] - yy[jj]))
        cov[ii][jj] = (njacks - 1)*cov[ii][jj]


yerr        = np.sqrt(np.diag(cov))

corr = 0.0*cov
for ii in range(len(xx)):
    for jj in range(len(xx)):
        corr[ii][jj] = cov[ii][jj]*1.0/(yerr[ii]*yerr[jj])


plt.subplot(2,2,1)
plt.imshow(corr,cmap='PuOr_r',vmin=-1,vmax=1,origin='lower',aspect='equal')
plt.xlabel(r'$R [h^{-1}{\rm Mpc }]$')
plt.ylabel(r'$R [h^{-1}{\rm Mpc }]$')
cb = plt.colorbar(fraction=0.046,pad=0.04)
cb.set_label(label=r"$r_{\rm ij}$")
plt.xticks(np.arange(10),np.round(xx, decimals=2),rotation=90)
plt.yticks(np.arange(10),np.round(xx, decimals=2))
plt.savefig('test_cov.png', dpi=300)

plt.clf()

plt.subplot(2,2,1)
plt.errorbar(xx, yy, yerr, fmt='.', capsize=3)
np.savetxt('./gal_proj.dat', np.transpose([xx, yy, yerr]))
plt.xlabel(r'$R\,[h^{-1}{\rm Mpc}]$')
plt.ylabel(r'$N(R)\,[h^{-1}{\rm Mpc}]^{-2}$')
#plt.ylabel(r'$\xi(R)$')
plt.xscale('log')
plt.yscale('log')
plt.savefig('test.png', dpi=300)

