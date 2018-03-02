from . import *

def Stokes(phi, D, U, rho=WATER_DENS, mu=WATER_DYN_VISC, norm=False):
    """Return Stokes force"""
    return 1 if norm else 9.42477796076938 * mu * D * U


def Ergun(phi, D, U, rho=WATER_DENS, mu=WATER_DYN_VISC, norm=True):
    """Calculate Ergun drag force per particle"""
    eps = 1. - phi
    Ergun = 0.0555555555555555 * (150 * (phi / eps**2) + 
            1.75 * (Reynolds(D, U, rho=rho, mu=mu) / eps**2))
#    for i in range(phi.shape[0]):
#        a = [phi[i], D[i], U[i], 
#             Reynolds(D[i], U[i], rho=rho, mu=mu),
#             (150 * (phi[i] / eps[i]**2)),
#             1.75 * (Reynolds(D[i], U[i], rho=rho, mu=mu) / eps[i]**2),
#             Ergun[i], Stokes(phi[i], D[i], U[i], rho=rho, mu=mu)/U[i],
#             Stokes(phi[i], D[i], U[i], rho=rho, mu=mu) * Ergun[i]]
#        print("Ergun: " + " ".join(["{0:0.9f}".format(j) for j in a]))
    return Ergun if norm else Stokes(phi, D, U, rho=rho, mu=mu) * Ergun


def DiFelice(phi, D, U, rho=WATER_DENS, mu=WATER_DYN_VISC, norm=True):
    """Calculate Di Felice drag force per particle"""
    Re = Reynolds(D, U, rho=rho, mu=mu)
    n = 3.7 - 0.65 * np.exp(-0.5 * (1.5 - np.log10(Re))**2)
    Cd = (0.63 + 4.8 * Re**-0.5)**2
    DiFelice = 0.75 * Cd * rho * U**2 * (1-phi)**-n * p_vol(D) / D
#    for i in range(phi.shape[0]):
#        a = [phi[i], Reynolds(D[i], U[i], rho=rho, mu=mu), 
#             Cd[i], n[i], p_vol(D[i]), (1-phi[i])**-n[i], D[i], phi[i], DiFelice[i]]
#        print("Di_Felice: " + " ".join(["{0:0.9f}".format(j) for j in a]))
    return DiFelice/ Stokes(phi, D, U, mu=mu) if norm else DiFelice


def Tang(phi, D, U, rho=WATER_DENS, mu=WATER_DYN_VISC, norm=True):
    """Calculate Tang drag force per particle"""
    eps = 1. - phi
    Re = Reynolds(D, U, rho=rho, mu=mu)
    Tang = (10 * phi / eps**2 
            + eps**2 * (1 + 1.5 * phi**0.5) 
            + Re * (  0.11 * phi * (1 + phi) 
                    - 0.00456 / eps**4 
                    + (0.169 * eps + 0.0644 / eps**4) 
                    * Re**(-0.343))) 
    return Tang if norm else Stokes(phi, D, U, mu=mu) * Tang


def Tenneti(phi, D, U, rho=WATER_DENS, mu=WATER_DYN_VISC, norm=True):
    """Calculate Tenneti drag force per particle"""
    eps = 1. - phi
    Re = Reynolds(D, U, rho=rho, mu=mu)
    Fd_0 = 0.44 * Re / 24. if Re > 1000 else 1. + 0.15 * Re**0.687
    Tenneti = Fd_0 / eps**2 + 5.81 * phi / eps**2 + 0.48 * phi**0.333333333 / eps**3 + \
            eps * phi**3 * Re * (0.95 + 0.61 * phi**3 / eps**2)
    return Tenneti if norm else Stokes(phi, D, U, mu=mu) * Tenneti


def BVK(phi, D, U, rho=WATER_DENS, mu=WATER_DYN_VISC, norm=True):
    """Calculate BVK drag force per particle"""
    eps = 1. - phi
    Re = Reynolds(D, U, rho=rho, mu=mu)
    BVK = (10 * phi / eps**2 
           + eps**2 * (1 + 1.5 * phi**0.5)
           + (0.413 * Re / (24 * eps**2)) 
           * (((1. / eps) + 3 * phi * eps + 8.4 * Re**-0.343) 
           / (1 + 10**(3*phi) * Re**-0.5*(1+4*phi))))
    return BVK if norm else Stokes(phi, D, U, mu=mu) * BVK

   
def BVK_poly(phi, diams, U, rho=WATER_DENS, mu=WATER_DYN_VISC, 
             norm=True, drag=BVK, nclasses=5):
    """Return total force on polydisperse assembly"""
    eps = 1. - phi
    #Mean volume weighted mean diameter sum(D*volume)/sum(volumes)
    mean_diam_g = calc_mean_diams(diams)[1]
    # Mean drag force
    mean_drag = drag(phi, mean_diam_g, U, rho=rho, mu=mu, norm=norm)
    # Histogram of indices of diameters
    bin_diams, bin_sizes = create_diam_bins(diams, nclasses)
    # Get average of histogram bin
    mean_diams = np.array([calc_mean_diams(diams[b])[1] for b in bin_diams])
    #Ratio of bin mean to average
    y1 = mean_diams / mean_diam_g
    Fp = mean_drag * (eps * y1 + phi * y1**2 + 0.064 * eps * y1**3)
    return np.dot(Fp, bin_sizes)
