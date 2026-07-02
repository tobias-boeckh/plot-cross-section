from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import unumpy as unp

from utils.colors import TangoColors
from utils.cross_section_wrapper import FaserCrossSectionWrapper
from utils.plot_utils import add_logo, save, show
from utils.utils import (
    get_figure_path,
    get_inv_qop_bins,
    get_log_mean,
    get_luminosity,
    get_num_qop_bins,
)

conv_fb_to_cm2 = 1.0e-39  # 1 fb = 1E-36 cm^2
scale_factor = 1.0e38


def get_eff_energies(
    eff_cross_section: np.ndarray | None = None,
    nu_nubar_ratio: float = 0.7684790543245343,
    num_steps: int = 10000,
) -> np.ndarray:
    """Get the energies, which correspond to the effective cross sections."""
    if eff_cross_section is None:
        eff_cross_section = np.load("data/xsec_eff.npy", allow_pickle=True)

    nu_energies = np.linspace(100, 2000, num_steps)
    wrapper = FaserCrossSectionWrapper()
    cross_section_nu = wrapper.get_cross_section(nu_energies, 14) * 1e-38
    cross_section_nubar = wrapper.get_cross_section(nu_energies, -14) * 1e-38
    cross_section_common = (
        nu_nubar_ratio * cross_section_nu + (1 - nu_nubar_ratio) * cross_section_nubar
    )

    # find energies where the cross section is closest to the simulated cross section
    eff_energies = np.zeros(get_num_qop_bins())
    for i in [0, 1, 2]:
        eff_energies[i] = nu_energies[
            np.argmin(
                np.abs(cross_section_nu - unp.nominal_values(eff_cross_section)[i])
            )
        ]
    for i in [4, 5]:
        eff_energies[i] = nu_energies[
            np.argmin(
                np.abs(cross_section_nubar - unp.nominal_values(eff_cross_section)[i])
            )
        ]
    eff_energies[3] = nu_energies[
        np.argmin(
            np.abs(cross_section_common - unp.nominal_values(eff_cross_section)[3])
        )
    ]
    return eff_energies


def get_data_xsec_plots(
    cross_section: np.ndarray,
    eff_energies: np.ndarray | None = None,
    energy_bins: np.ndarray | None = None,
    scaling: float = 1e-38,
    indices: list[int] | None = None,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    markers: list[str] | None = None,
    markersize_scalings: list[float] | None = None,
    asimov_data: bool = False,
    normalize_energy: bool = True,
) -> list[dict[str, Any]]:
    if indices is None:
        indices = [[0, 1, 2], [4, 5], [3]]
    if labels is None:
        labels = [
            r"$\nu_{\mu}$",
            r"$\bar{\nu_{\mu}}$",
            r"$\nu_{\mu} + \bar{\nu_{\mu}}$",
        ]
        if asimov_data:
            labels = [f"{label} (Asimov)" for label in labels]
    if colors is None:
        colors = [TangoColors.scarlet_red_dark] * len(indices)
    if markers is None:
        markers = ["o", "*", "s"]
    if markersize_scalings is None:
        markersize_scalings = [1.0, 1.5, 1.0]
    if eff_energies is None:
        eff_energies = get_eff_energies()
    if normalize_energy:
        _cross_section = cross_section / unp.nominal_values(eff_energies)
    else:
        _cross_section = cross_section

    if energy_bins is None:
        energy_bins = get_inv_qop_bins()
    centers = 0.5 * (energy_bins[:, 0] + energy_bins[:, 1])
    lower = centers - energy_bins[:, 0]
    upper = energy_bins[:, 1] - centers
    plots = [
        {
            "E": centers[idx],
            "E_err": (lower[idx], upper[idx]),
            "xsec": unp.nominal_values(_cross_section)[idx] / scaling,
            "xsec_err": unp.std_devs(_cross_section)[idx] / scaling,
            "label": label,
            "color": color,
            "marker": marker,
            "markersize_scaling": markersize_scaling,
        }
        for idx, label, color, marker, markersize_scaling in zip(
            indices, labels, colors, markers, markersize_scalings
        )
    ]
    return plots


def get_bodek_yang_plots(
    lower: float = 100,
    upper: float = 10000,
    nu_nubar_ratio: float = 0.7684790543245343,
    num_steps: int = 10000,
    scaling: float = 1e-38,
    colors: list[str] | None = None,
    linestyles: list[str] | None = None,
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    nu_energies = np.linspace(lower, upper, num_steps)
    wrapper = FaserCrossSectionWrapper()
    cross_section_nu = wrapper.get_cross_section(nu_energies, 14) / scaling
    cross_section_nubar = wrapper.get_cross_section(nu_energies, -14) / scaling

    cross_sections = [
        cross_section_nu,
        cross_section_nubar,
        cross_section_nu * nu_nubar_ratio + cross_section_nubar * (1 - nu_nubar_ratio),
    ]
    if linestyles is None:
        # linestyles = ["--", "--", "--"]
        linestyles = ["--", ":", "-."]
    if colors is None:
        colors = ["gainsboro"] * 3
    if labels is None:
        # labels = [""] * 3
        labels = [
            r"Bodek-Yang sim. $\nu_{\mu}$",
            r"Bodek-Yang sim. $\bar{\nu}_{\mu}$",
            r"Bodek-Yang sim. $\nu_{\mu} + \bar{\nu}_{\mu}$",
        ]

    bodek_yang_plots = [
        {
            "E": nu_energies,
            "xsec": cross_section * scaling,
            "ls": linestyle,
            "label": label,
            "color": color,
        }
        for cross_section, linestyle, label, color in zip(
            cross_sections, linestyles, labels, colors
        )
    ]
    return bodek_yang_plots


def get_cross_section_nu_nubar(
    lower: float = 100,
    upper: float = 10000,
    nu_nubar_ratio: float = 0.7684790543245343,
    num_steps: int = 10000,
    scaling: float = 1e-38,
) -> np.ndarray:
    nu_energies = np.linspace(lower, upper, num_steps)
    wrapper = FaserCrossSectionWrapper()
    cross_section_nu = wrapper.get_cross_section(nu_energies, 14) / scaling
    cross_section_nubar = wrapper.get_cross_section(nu_energies, -14) / scaling
    cross_section_nu_nubar = 0.5 * (
        cross_section_nu * nu_nubar_ratio + cross_section_nubar * (1 - nu_nubar_ratio)
    )
    return cross_section_nu_nubar


def Plot_Cross_Section_Measurements(
    ax,
    markersize=5,
    experiments=None,
    colors=None,
    print_xsref=True,
    divide_energy=True,
    elinewidth: float = 1.0,
    curved_fasernu_errorbar: bool = True,
):
    #################################################################################
    # T2K (C)
    # PRD 92, 112003 (2015)
    # https://www.hepdata.net/record/ins1329784

    r"""
    # Table 1 (NUMU)
    # '$E_\nu$ [GeV]','$E_\nu$ [GeV] LOW','$E_\nu$ [GeV] HIGH',r'$\sigma(E)$ [$10^{-38}$ cm$^{2}$]','error +','error -'
    # 0.43761,0.0,0.6,0.68491,0.13303,-0.11087
    # 0.6482,0.6,0.7,0.7715,0.2071,-0.18971
    # 0.8091,0.7,1.0,1.08959,0.23881,-0.19218
    # 1.19633,1.0,1.5,0.3689,0.34782,-0.32917
    # 3.37397,1.5,30.0,1.37898,0.43842,-0.40189
    """
    arr_raw = [
        [0.43761, 0.0, 0.6, 0.68491, 0.13303, -0.11087],
        [0.6482, 0.6, 0.7, 0.7715, 0.2071, -0.18971],
        [1.19633, 1.0, 1.5, 0.3689, 0.34782, -0.32917],
        [3.37397, 1.5, 30.0, 1.37898, 0.43842, -0.40189],
    ]

    arr_cs_cc_numu_t2k_c_2015_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_t2k_c_2015_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_t2k_c_2015_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_t2k_c_2015_cs_central = [x[3] / x[0] for x in arr_raw]
        arr_cs_cc_numu_t2k_c_2015_cs_hi = [x[4] / x[0] for x in arr_raw]
        arr_cs_cc_numu_t2k_c_2015_cs_lo = [-x[5] / x[0] for x in arr_raw]
    else:
        arr_cs_cc_numu_t2k_c_2015_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_t2k_c_2015_cs_hi = [x[4] for x in arr_raw]
        arr_cs_cc_numu_t2k_c_2015_cs_lo = [-x[5] for x in arr_raw]

    # There is no NUMUBAR data
    #################################################################################

    #################################################################################
    # T2K (C)
    # PRD 87, 092003 (2013)
    # https://arxiv.org/abs/1302.4908

    r"""
        "# Eq. (12)
        # '$E_\nu$ [GeV]','$E_\nu$ [GeV] LOW','$E_\nu$ [GeV] HIGH',r'$\sigma(E)$ [$10^{-38}$ cm$^{2}$]','error +','error -','sys error +'.'sys error -'
    """

    arr_raw = [
        [0.85, 0.58, 1.25, 0.691, 0.013, 0.013, 0.084, 0.084],
    ]

    arr_cs_cc_numu_t2k_c_2013_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_t2k_c_2013_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_t2k_c_2013_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_t2k_c_2013_cs_central = [x[3] / x[0] for x in arr_raw]
        arr_cs_cc_numu_t2k_c_2013_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) / x[0] for x in arr_raw
        ]
        arr_cs_cc_numu_t2k_c_2013_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) / x[0] for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_t2k_c_2013_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_t2k_c_2013_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) for x in arr_raw
        ]
        arr_cs_cc_numu_t2k_c_2013_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) for x in arr_raw
        ]

    # There is no NUMUBAR data
    #################################################################################

    #################################################################################
    # T2K (Fe)
    # PRD 90, 052010 (2014)
    # https://arxiv.org/abs/1407.4256

    r"""
        # Eq. (2)
        # '$E_\nu$ [GeV]','$E_\nu$ [GeV] LOW','$E_\nu$ [GeV] HIGH',r'$\sigma(E)$ [$10^{-38}$ cm$^{2}$]','error +','error -','sys error +'.'sys error -'
    """

    arr_raw = [
        [
            1.51,
            0.6965594138260591,
            2.3129181267919714,
            1.444,
            0.002,
            0.002,
            0.189,
            0.157,
        ],
    ]

    arr_cs_cc_numu_t2k_fe_2014_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_t2k_fe_2014_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_t2k_fe_2014_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_t2k_fe_2014_cs_central = [x[3] / x[0] for x in arr_raw]
        arr_cs_cc_numu_t2k_fe_2014_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) / x[0] for x in arr_raw
        ]
        arr_cs_cc_numu_t2k_fe_2014_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) / x[0] for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_t2k_fe_2014_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_t2k_fe_2014_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) for x in arr_raw
        ]
        arr_cs_cc_numu_t2k_fe_2014_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) for x in arr_raw
        ]

    # There is no NUMUBAR data
    #################################################################################

    #################################################################################
    # T2K (CH)
    # PRD 90, 052010 (2014)
    # https://arxiv.org/abs/1407.4256

    r"""
        # Eq. (2)
        # '$E_\nu$ [GeV]','$E_\nu$ [GeV] LOW','$E_\nu$ [GeV] HIGH',r'$\sigma(E)$ [$10^{-38}$ cm$^{2}$]','error +','error -','sys error +'.'sys error -'
    """

    arr_raw = [
        [1.51, 0.7368421052631577, 2.35672514619883, 1.379, 0.009, 0.009, 0.178, 0.147],
    ]

    arr_cs_cc_numu_t2k_ch_2014_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_t2k_ch_2014_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_t2k_ch_2014_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_t2k_ch_2014_cs_central = [x[3] / x[0] for x in arr_raw]
        arr_cs_cc_numu_t2k_ch_2014_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) / x[0] for x in arr_raw
        ]
        arr_cs_cc_numu_t2k_ch_2014_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) / x[0] for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_t2k_ch_2014_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_t2k_ch_2014_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) for x in arr_raw
        ]
        arr_cs_cc_numu_t2k_ch_2014_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) for x in arr_raw
        ]
    # There is no NUMUBAR data
    #################################################################################

    #################################################################################
    # IHEP-ITEP
    # Sov. J. Nucl. Phys. 30 (1979) 528, 1979
    # https://www.hepdata.net/record/ins141742

    """
        # Table 1 (NUMU)
        # 'E [GEV]','E [GEV] LOW','E [GEV] HIGH','SIG/E [FB/GEV]','error +','error -','sys,DUE TO THE NORMALISATION ERROR +','sys,DUE TO THE NORMALISATION ERROR -'
        # 6.5,5.0,8.0,7.7,0.4,-0.4,'7.0%','-7.0%'
        # 10.0,8.0,12.0,7.4,0.4,-0.4,'7.0%','-7.0%'
        # 16.0,12.0,20.0,7.2,0.4,-0.4,'7.0%','-7.0%'
        # 27.5,20.0,35.0,6.8,0.6,-0.6,'7.0%','-7.0%'
    """
    arr_raw = [
        [6.5, 5.0, 8.0, 7.7, 0.4, -0.4, "7.0%", "-7.0%"],
        [10.0, 8.0, 12.0, 7.4, 0.4, -0.4, "7.0%", "-7.0%"],
        [16.0, 12.0, 20.0, 7.2, 0.4, -0.4, "7.0%", "-7.0%"],
        [27.5, 20.0, 35.0, 6.8, 0.6, -0.6, "7.0%", "-7.0%"],
    ]

    arr_cs_cc_numu_ihep_itep_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_ihep_itep_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_ihep_itep_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_ihep_itep_cs_central = [
            x[3] * conv_fb_to_cm2 * scale_factor for x in arr_raw
        ]
        arr_cs_cc_numu_ihep_itep_cs_hi = [
            x[4] * conv_fb_to_cm2 * scale_factor for x in arr_raw
        ]
        arr_cs_cc_numu_ihep_itep_cs_lo = [
            -x[5] * conv_fb_to_cm2 * scale_factor for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_ihep_itep_cs_central = [
            x[3] * conv_fb_to_cm2 * scale_factor * x[0] for x in arr_raw
        ]
        arr_cs_cc_numu_ihep_itep_cs_hi = [
            x[4] * conv_fb_to_cm2 * scale_factor * x[0] for x in arr_raw
        ]
        arr_cs_cc_numu_ihep_itep_cs_lo = [
            -x[5] * conv_fb_to_cm2 * scale_factor * x[0] for x in arr_raw
        ]
    """
        # Table 2 (NUMUBAR)
        # 'E [GEV]','E [GEV] LOW','E [GEV] HIGH','SIG/E [FB/GEV]','error +','error -','sys,DUE TO THE NORMALISATION ERROR +','sys,DUE TO THE NORMALISATION ERROR -'
        # 6.5,5.0,8.0,3.0,0.2,-0.2,'7.0%','-7.0%'
        # 10.0,8.0,12.0,3.1,0.3,-0.3,'7.0%','-7.0%'
        # 16.0,12.0,20.0,3.2,0.4,-0.4,'7.0%','-7.0%'
        # 27.5,20.0,35.0,3.2,0.6,-0.6,'7.0%','-7.0%'
    """
    arr_raw = [
        [6.5, 5.0, 8.0, 3.0, 0.2, -0.2, "7.0%", "-7.0%"],
        [10.0, 8.0, 12.0, 3.1, 0.3, -0.3, "7.0%", "-7.0%"],
        [16.0, 12.0, 20.0, 3.2, 0.4, -0.4, "7.0%", "-7.0%"],
        [27.5, 20.0, 35.0, 3.2, 0.6, -0.6, "7.0%", "-7.0%"],
    ]

    arr_cs_cc_numubar_ihep_itep_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numubar_ihep_itep_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numubar_ihep_itep_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numubar_ihep_itep_cs_central = [
            x[3] * conv_fb_to_cm2 * scale_factor for x in arr_raw
        ]
        arr_cs_cc_numubar_ihep_itep_cs_hi = [
            x[4] * conv_fb_to_cm2 * scale_factor for x in arr_raw
        ]
        arr_cs_cc_numubar_ihep_itep_cs_lo = [
            -x[5] * conv_fb_to_cm2 * scale_factor for x in arr_raw
        ]
    else:
        arr_cs_cc_numubar_ihep_itep_cs_central = [
            x[3] * conv_fb_to_cm2 * scale_factor * x[0] for x in arr_raw
        ]
        arr_cs_cc_numubar_ihep_itep_cs_hi = [
            x[4] * conv_fb_to_cm2 * scale_factor * x[0] for x in arr_raw
        ]
        arr_cs_cc_numubar_ihep_itep_cs_lo = [
            -x[5] * conv_fb_to_cm2 * scale_factor * x[0] for x in arr_raw
        ]
    #################################################################################

    #################################################################################
    # NOMAD
    # Phys. Lett. B660 (2008) 19-25, 2008
    # https://hepdata.net/record/ins767013

    """
        # Table 1 (NUMU)
        # 'E(P=1) [GEV]','E(P=1) [GEV] LOW','E(P=1) [GEV] HIGH','SIG/E(P=1) [10**-38CM**2/GEV]','stat +','stat -','sys +','sys -'
        # 4.6,2.5,6.0,0.786,0.011,-0.011,0.035,-0.035
        # 6.5,6.0,7.0,0.763,0.011,-0.011,0.036,-0.036
        # 7.5,7.0,8.0,0.722,0.009,-0.009,0.035,-0.035
        # 8.5,8.0,9.0,0.701,0.007,-0.007,0.033,-0.033
        # 9.5,9.0,10.0,0.716,0.007,-0.007,0.033,-0.033
        # 10.5,10.0,11.0,0.706,0.005,-0.005,0.026,-0.026
        # 11.5,11.0,12.0,0.705,0.005,-0.005,0.024,-0.024
        # 12.5,12.0,13.0,0.697,0.005,-0.005,0.024,-0.024
        # 13.5,13.0,14.0,0.7,0.005,-0.005,0.024,-0.024
        # 14.5,14.0,15.0,0.698,0.004,-0.004,0.025,-0.025
        # 16.2,15.0,17.5,0.698,0.003,-0.003,0.025,-0.025
        # 18.7,17.5,20.0,0.7,0.003,-0.003,0.025,-0.025
        # 21.2,20.0,22.5,0.699,0.003,-0.003,0.024,-0.024
        # 23.7,22.5,25.0,0.694,0.003,-0.003,0.024,-0.024
        # 26.2,25.0,27.5,0.694,0.003,-0.003,0.025,-0.025
        # 28.7,27.5,30.0,0.694,0.003,-0.003,0.025,-0.025
        # 32.3,30.0,35.0,0.677,0.003,-0.003,0.026,-0.026
        # 37.3,35.0,40.0,0.681,0.003,-0.003,0.026,-0.026
        # 42.4,40.0,45.0,0.675,0.003,-0.003,0.028,-0.028
        # 47.4,45.0,50.0,0.682,0.004,-0.004,0.027,-0.027
        # 54.6,50.0,60.0,0.67,0.003,-0.003,0.028,-0.028
        # 64.7,60.0,70.0,0.675,0.003,-0.003,0.031,-0.031
        # 74.8,70.0,80.0,0.684,0.003,-0.003,0.037,-0.037
        # 84.8,80.0,90.0,0.678,0.004,-0.004,0.041,-0.041
        # 94.8,90.0,100.0,0.677,0.004,-0.004,0.043,-0.043
        # 107.0,100.0,115.0,0.674,0.004,-0.004,0.048,-0.048
        # 122.0,115.0,130.0,0.661,0.005,-0.005,0.048,-0.048
        # 136.9,130.0,145.0,0.671,0.006,-0.006,0.054,-0.054
        # 165.9,145.0,200.0,0.667,0.004,-0.004,0.054,-0.054
        # 228.3,200.0,300.0,0.721,0.008,-0.008,0.06,-0.06
    """

    arr_raw = [
        [4.6, 2.5, 6.0, 0.786, 0.011, -0.011, 0.035, -0.035],
        [6.5, 6.0, 7.0, 0.763, 0.011, -0.011, 0.036, -0.036],
        [7.5, 7.0, 8.0, 0.722, 0.009, -0.009, 0.035, -0.035],
        [8.5, 8.0, 9.0, 0.701, 0.007, -0.007, 0.033, -0.033],
        [9.5, 9.0, 10.0, 0.716, 0.007, -0.007, 0.033, -0.033],
        [10.5, 10.0, 11.0, 0.706, 0.005, -0.005, 0.026, -0.026],
        [11.5, 11.0, 12.0, 0.705, 0.005, -0.005, 0.024, -0.024],
        [12.5, 12.0, 13.0, 0.697, 0.005, -0.005, 0.024, -0.024],
        [13.5, 13.0, 14.0, 0.7, 0.005, -0.005, 0.024, -0.024],
        [14.5, 14.0, 15.0, 0.698, 0.004, -0.004, 0.025, -0.025],
        [16.2, 15.0, 17.5, 0.698, 0.003, -0.003, 0.025, -0.025],
        [18.7, 17.5, 20.0, 0.7, 0.003, -0.003, 0.025, -0.025],
        [21.2, 20.0, 22.5, 0.699, 0.003, -0.003, 0.024, -0.024],
        [23.7, 22.5, 25.0, 0.694, 0.003, -0.003, 0.024, -0.024],
        [26.2, 25.0, 27.5, 0.694, 0.003, -0.003, 0.025, -0.025],
        [28.7, 27.5, 30.0, 0.694, 0.003, -0.003, 0.025, -0.025],
        [32.3, 30.0, 35.0, 0.677, 0.003, -0.003, 0.026, -0.026],
        [37.3, 35.0, 40.0, 0.681, 0.003, -0.003, 0.026, -0.026],
        [42.4, 40.0, 45.0, 0.675, 0.003, -0.003, 0.028, -0.028],
        [47.4, 45.0, 50.0, 0.682, 0.004, -0.004, 0.027, -0.027],
        [54.6, 50.0, 60.0, 0.67, 0.003, -0.003, 0.028, -0.028],
        [64.7, 60.0, 70.0, 0.675, 0.003, -0.003, 0.031, -0.031],
        [74.8, 70.0, 80.0, 0.684, 0.003, -0.003, 0.037, -0.037],
        [84.8, 80.0, 90.0, 0.678, 0.004, -0.004, 0.041, -0.041],
        [94.8, 90.0, 100.0, 0.677, 0.004, -0.004, 0.043, -0.043],
        [107.0, 100.0, 115.0, 0.674, 0.004, -0.004, 0.048, -0.048],
        [122.0, 115.0, 130.0, 0.661, 0.005, -0.005, 0.048, -0.048],
        [136.9, 130.0, 145.0, 0.671, 0.006, -0.006, 0.054, -0.054],
        [165.9, 145.0, 200.0, 0.667, 0.004, -0.004, 0.054, -0.054],
        [228.3, 200.0, 300.0, 0.721, 0.008, -0.008, 0.06, -0.06],
    ]

    arr_cs_cc_numu_nomad_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_nomad_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_nomad_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_nomad_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_nomad_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) for x in arr_raw
        ]
        arr_cs_cc_numu_nomad_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_nomad_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numu_nomad_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) * x[0] for x in arr_raw
        ]
        arr_cs_cc_numu_nomad_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) * x[0] for x in arr_raw
        ]
    # There is no NUMUBAR data
    #################################################################################

    #################################################################################
    # MINOS
    # Phys. Rev. D81 (2010) 072002
    # https://arxiv.org/abs/0910.2201

    """
        # Table III (NUMU)
        # 'E [GEV]','E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -','sys ERROR +','sys ERROR -','sys,DUE TO THE NORMALISATION ERROR +','sys,DUE TO THE NORMALISATION ERROR -', 'total error+', total error-'
    """

    arr_raw = [
        [3.48, 3.0, 4.0, 0.748, 0.003, 0.003, 0.058, 0.058, 0.017, 0.017, 0.061, 0.061],
        [4.45, 4.0, 5.0, 0.711, 0.004, 0.004, 0.029, 0.029, 0.017, 0.017, 0.033, 0.033],
        [5.89, 5.0, 7.0, 0.708, 0.005, 0.005, 0.027, 0.027, 0.016, 0.016, 0.032, 0.032],
        [7.97, 7.0, 9.0, 0.722, 0.006, 0.006, 0.041, 0.041, 0.017, 0.017, 0.045, 0.045],
        [
            10.45,
            9.0,
            12.0,
            0.699,
            0.005,
            0.005,
            0.041,
            0.041,
            0.014,
            0.014,
            0.043,
            0.043,
        ],
        [
            13.43,
            12.0,
            15.0,
            0.691,
            0.006,
            0.006,
            0.023,
            0.023,
            0.014,
            0.014,
            0.028,
            0.028,
        ],
        [
            16.42,
            15.0,
            18.0,
            0.708,
            0.008,
            0.008,
            0.012,
            0.012,
            0.014,
            0.014,
            0.020,
            0.020,
        ],
        [
            19.87,
            18.0,
            22.0,
            0.689,
            0.006,
            0.006,
            0.009,
            0.009,
            0.012,
            0.012,
            0.016,
            0.016,
        ],
        [
            23.88,
            22.0,
            26.0,
            0.683,
            0.008,
            0.008,
            0.005,
            0.005,
            0.012,
            0.012,
            0.015,
            0.015,
        ],
        [
            27.89,
            26.0,
            30.0,
            0.686,
            0.010,
            0.010,
            0.004,
            0.004,
            0.012,
            0.012,
            0.016,
            0.016,
        ],
        [
            32.81,
            30.0,
            36.0,
            0.675,
            0.010,
            0.010,
            0.002,
            0.002,
            0.011,
            0.011,
            0.016,
            0.016,
        ],
        [
            38.87,
            36.0,
            42.0,
            0.675,
            0.013,
            0.013,
            0.005,
            0.005,
            0.011,
            0.011,
            0.018,
            0.018,
        ],
        [
            45.77,
            42.0,
            50.0,
            0.676,
            0.014,
            0.014,
            0.004,
            0.004,
            0.011,
            0.011,
            0.019,
            0.019,
        ],
    ]

    arr_cs_cc_numu_minos_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_minos_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_minos_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_minos_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_minos_cs_hi = [x[10] for x in arr_raw]
        arr_cs_cc_numu_minos_cs_lo = [x[11] for x in arr_raw]
    else:
        arr_cs_cc_numu_minos_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numu_minos_cs_hi = [x[10] * x[0] for x in arr_raw]
        arr_cs_cc_numu_minos_cs_lo = [x[11] * x[0] for x in arr_raw]
    """
        # Table III (NUMUBAR)
        # 'E [GEV]','E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -','sys ERROR +','sys ERROR -','sys,DUE TO THE NORMALISATION ERROR +','sys,DUE TO THE NORMALISATION ERROR -', 'total error+', total error-'
    """

    arr_raw = [
        [6.07, 5.0, 7.0, 0.305, 0.005, 0.005, 0.027, 0.027, 0.007, 0.007, 0.029, 0.029],
        [7.99, 7.0, 9.0, 0.300, 0.005, 0.005, 0.021, 0.021, 0.007, 0.007, 0.022, 0.022],
        [
            10.43,
            9.0,
            12.0,
            0.303,
            0.004,
            0.004,
            0.018,
            0.018,
            0.006,
            0.006,
            0.019,
            0.019,
        ],
        [
            13.42,
            12.0,
            15.0,
            0.314,
            0.005,
            0.005,
            0.014,
            0.014,
            0.006,
            0.006,
            0.016,
            0.016,
        ],
        [
            16.41,
            15.0,
            18.0,
            0.304,
            0.007,
            0.007,
            0.007,
            0.007,
            0.006,
            0.006,
            0.012,
            0.012,
        ],
        [
            19.82,
            18.0,
            22.0,
            0.316,
            0.006,
            0.006,
            0.011,
            0.011,
            0.005,
            0.005,
            0.013,
            0.013,
        ],
        [
            23.82,
            22.0,
            26.0,
            0.320,
            0.009,
            0.009,
            0.004,
            0.004,
            0.005,
            0.005,
            0.011,
            0.011,
        ],
        [
            27.84,
            26.0,
            30.0,
            0.332,
            0.012,
            0.012,
            0.005,
            0.005,
            0.006,
            0.006,
            0.015,
            0.015,
        ],
        [
            32.72,
            30.0,
            36.0,
            0.325,
            0.014,
            0.014,
            0.006,
            0.006,
            0.005,
            0.005,
            0.016,
            0.016,
        ],
        [
            38.74,
            36.0,
            42.0,
            0.352,
            0.021,
            0.021,
            0.011,
            0.011,
            0.006,
            0.006,
            0.024,
            0.024,
        ],
        [
            45.61,
            42.0,
            50.0,
            0.324,
            0.023,
            0.023,
            0.013,
            0.013,
            0.005,
            0.005,
            0.027,
            0.027,
        ],
    ]

    arr_cs_cc_numubar_minos_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numubar_minos_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numubar_minos_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numubar_minos_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numubar_minos_cs_hi = [x[10] for x in arr_raw]
        arr_cs_cc_numubar_minos_cs_lo = [x[11] for x in arr_raw]
    else:
        arr_cs_cc_numubar_minos_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numubar_minos_cs_hi = [x[10] * x[0] for x in arr_raw]
        arr_cs_cc_numubar_minos_cs_lo = [x[11] * x[0] for x in arr_raw]
    #################################################################################

    #################################################################################
    # SKAT
    # Phys. Lett. B81 (1979) 255
    # http://www.sciencedirect.com/science/article/pii/0370269379905367?via%3Dihub

    """
        # Table 1 (NUMU)
        # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [
        [3.0, 5.0, 0.68, 0.07, 0.07],
        [5.0, 9.0, 0.70, 0.09, 0.09],
        [9.0, 16.0, 0.86, 0.13, 0.13],
        [16.0, 30.0, 0.70, 0.15, 0.15],
    ]

    arr_cs_cc_numu_skat_energy_central = [(x[0] + x[1]) / 2.0 for x in arr_raw]
    arr_cs_cc_numu_skat_energy_lo = [x[0] - (x[0] + x[1]) / 2.0 for x in arr_raw]
    arr_cs_cc_numu_skat_energy_hi = [(x[0] + x[1]) / 2.0 - x[1] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_skat_cs_central = [x[2] for x in arr_raw]
        arr_cs_cc_numu_skat_cs_hi = [x[3] for x in arr_raw]
        arr_cs_cc_numu_skat_cs_lo = [x[4] for x in arr_raw]
    else:
        arr_cs_cc_numu_skat_cs_central = [x[2] * (x[0] + x[1]) / 2.0 for x in arr_raw]
        arr_cs_cc_numu_skat_cs_hi = [x[3] * (x[0] + x[1]) / 2.0 for x in arr_raw]
        arr_cs_cc_numu_skat_cs_lo = [x[4] * (x[0] + x[1]) / 2.0 for x in arr_raw]

    # There is no NUMUBAR data
    #################################################################################

    #################################################################################
    # SciBooNE
    # Phys. Rev. D83 (2011) 012005
    # https://arxiv.org/abs/1011.2131

    """
        # Table XII (NUMU) -- NUANCE-based
        # 'E [GEV]','E [GEV] LOW','E [GEV] HIGH','SIG [10**-38CM**2]','error +','error -'
    """
    arr_raw = [
        [0.38, 0.25, 0.50, 0.340, 0.096, 0.096],
        [0.62, 0.50, 0.75, 0.639, 0.081, 0.081],
        [0.87, 0.75, 1.00, 1.01, 0.09, 0.09],
        [1.11, 1.00, 1.25, 1.29, 0.15, 0.15],
        [1.43, 1.25, 1.75, 1.56, 0.28, 0.28],
        [2.47, 1.75, 3.00, 1.66, 0.37, 0.37],
    ]

    arr_cs_cc_numu_sciboone_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_sciboone_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_sciboone_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_sciboone_cs_central = [x[3] / x[0] for x in arr_raw]
        arr_cs_cc_numu_sciboone_cs_hi = [x[4] / x[0] for x in arr_raw]
        arr_cs_cc_numu_sciboone_cs_lo = [x[5] / x[0] for x in arr_raw]
    else:
        arr_cs_cc_numu_sciboone_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_sciboone_cs_hi = [x[4] for x in arr_raw]
        arr_cs_cc_numu_sciboone_cs_lo = [x[5] for x in arr_raw]

    # There is no NUMUBAR data
    #################################################################################

    #################################################################################
    # GGM-PS
    # Phys. Lett. B84 (1979) 281-284, 1979
    # https://hepdata.net/record/ins141175

    """
    # Table 1 (NUMU)
    # 'PLAB [GEV]','PLAB [GEV] LOW','PLAB [GEV] HIGH','SIG/E [10**-38 CM**2/GEV]','error +','error -'
    # 2.87,1.5,15.0,0.69,0.05,-0.05
    # 9.05,6.0,15.0,0.61,0.06,-0.06
    """
    arr_raw = [
        [2.87, 1.5, 15.0, 0.69, 0.05, -0.05],
        [9.05, 6.0, 15.0, 0.61, 0.06, -0.06],
    ]

    arr_cs_cc_numu_ggm_ps_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_ggm_ps_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_ggm_ps_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_ggm_ps_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_ggm_ps_cs_hi = [x[4] for x in arr_raw]
        arr_cs_cc_numu_ggm_ps_cs_lo = [-x[5] for x in arr_raw]
    else:
        arr_cs_cc_numu_ggm_ps_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numu_ggm_ps_cs_hi = [x[4] * x[0] for x in arr_raw]
        arr_cs_cc_numu_ggm_ps_cs_lo = [-x[5] * x[0] for x in arr_raw]
    """
    # Table 2 (NUMUBAR)
    # 'PLAB [GEV]','PLAB [GEV] LOW','PLAB [GEV] HIGH','SIG/E [10**-38 CM**2/GEV]','error +','error -'
    # 3.0,1.5,15.0,0.26,0.02,-0.02
    """
    arr_raw = [[3.0, 1.5, 15.0, 0.26, 0.02, -0.02]]

    arr_cs_cc_numubar_ggm_ps_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numubar_ggm_ps_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numubar_ggm_ps_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numubar_ggm_ps_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numubar_ggm_ps_cs_hi = [x[4] for x in arr_raw]
        arr_cs_cc_numubar_ggm_ps_cs_lo = [-x[5] for x in arr_raw]
    else:
        arr_cs_cc_numubar_ggm_ps_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numubar_ggm_ps_cs_hi = [x[4] * x[0] for x in arr_raw]
        arr_cs_cc_numubar_ggm_ps_cs_lo = [-x[5] * x[0] for x in arr_raw]
    #################################################################################

    #################################################################################
    # ANL
    # Phys. Rev. D19 (1979) 2521, 1979
    # https://hepdata.net/record/ins7237

    """
    # Table 1 (NUMU)
    # 'PLAB [GEV]','PLAB [GEV] LOW','PLAB [GEV] HIGH','SIG/E [10**-38 CM**2/GEV]','error +','error -'
    # 3.1,0.2,6.0,0.87,0.03,-0.03
    # 1.1,1.1,1.1,0.76,0.06,-0.06
    """

    arr_raw = [[3.1, 0.2, 6.0, 0.87, 0.03, -0.03], [1.1, 1.1, 1.1, 0.76, 0.06, -0.06]]

    arr_cs_cc_numu_anl_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_anl_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_anl_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_anl_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_anl_cs_hi = [x[4] for x in arr_raw]
        arr_cs_cc_numu_anl_cs_lo = [-x[5] for x in arr_raw]
    else:
        arr_cs_cc_numu_anl_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numu_anl_cs_hi = [x[4] * x[0] for x in arr_raw]
        arr_cs_cc_numu_anl_cs_lo = [-x[5] * x[0] for x in arr_raw]

    # There is no NUMUBAR data
    #################################################################################

    #################################################################################
    # BEBC
    # Z. Phys. C1 (1979) 143
    # http://dx.doi.org/10.1007/BF01445406

    """
    # Table 2 (NUMU)
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [[30.0, 90.0, 0.62, 0.04, 0.04], [90.0, 190.0, 0.63, 0.05, 0.05]]

    arr_cs_cc_numu_bebc_1_energy_central = [(x[0] + x[1]) / 2.0 for x in arr_raw]
    arr_cs_cc_numu_bebc_1_energy_lo = [(x[0] + x[1]) / 2.0 - x[0] for x in arr_raw]
    arr_cs_cc_numu_bebc_1_energy_hi = [x[1] - (x[0] + x[1]) / 2.0 for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_bebc_1_cs_central = [x[2] for x in arr_raw]
        arr_cs_cc_numu_bebc_1_cs_hi = [x[3] for x in arr_raw]
        arr_cs_cc_numu_bebc_1_cs_lo = [x[4] for x in arr_raw]
    else:
        arr_cs_cc_numu_bebc_1_cs_central = [x[2] * (x[0] + x[1]) / 2.0 for x in arr_raw]
        arr_cs_cc_numu_bebc_1_cs_hi = [x[3] * (x[0] + x[1]) / 2.0 for x in arr_raw]
        arr_cs_cc_numu_bebc_1_cs_lo = [x[4] * (x[0] + x[1]) / 2.0 for x in arr_raw]

    """
    # Table 2 (NUMUBAR)
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [[30.0, 90.0, 0.30, 0.02, 0.02], [90.0, 190.0, 0.31, 0.04, 0.04]]

    arr_cs_cc_numubar_bebc_1_energy_central = [(x[0] + x[1]) / 2.0 for x in arr_raw]
    arr_cs_cc_numubar_bebc_1_energy_lo = [(x[0] + x[1]) / 2.0 - x[0] for x in arr_raw]
    arr_cs_cc_numubar_bebc_1_energy_hi = [x[1] - (x[0] + x[1]) / 2.0 for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numubar_bebc_1_cs_central = [x[2] for x in arr_raw]
        arr_cs_cc_numubar_bebc_1_cs_hi = [x[3] for x in arr_raw]
        arr_cs_cc_numubar_bebc_1_cs_lo = [x[4] for x in arr_raw]
    else:
        arr_cs_cc_numubar_bebc_1_cs_central = [
            x[2] * (x[0] + x[1]) / 2.0 for x in arr_raw
        ]
        arr_cs_cc_numubar_bebc_1_cs_hi = [x[3] * (x[0] + x[1]) / 2.0 for x in arr_raw]
        arr_cs_cc_numubar_bebc_1_cs_lo = [x[4] * (x[0] + x[1]) / 2.0 for x in arr_raw]
    #################################################################################

    #################################################################################
    # BEBC
    # Z. Phys. C2 (1979) 187
    # http://dx.doi.org/10.1007/BF01474659

    """
    # Taken from Table 1 of Z. Phys. C 70 (1996) 39 (NUMU)
    # 'E [GEV]','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [[19.0, 0.74, 0.11, 0.11], [34.0, 0.73, 0.10, 0.10]]

    arr_cs_cc_numu_bebc_2_energy_central = [x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_bebc_2_cs_central = [x[1] for x in arr_raw]
        arr_cs_cc_numu_bebc_2_cs_hi = [x[2] for x in arr_raw]
        arr_cs_cc_numu_bebc_2_cs_lo = [x[3] for x in arr_raw]
    else:
        arr_cs_cc_numu_bebc_2_cs_central = [x[1] * x[0] for x in arr_raw]
        arr_cs_cc_numu_bebc_2_cs_hi = [x[2] * x[0] for x in arr_raw]
        arr_cs_cc_numu_bebc_2_cs_lo = [x[3] * x[0] for x in arr_raw]

    # There is no NUMUBAR data
    #################################################################################

    #################################################################################
    # IHEP-JINR
    # Z. Phys. C70 (1996) 39
    # http://dx.doi.org/10.1007/s002880050078

    """
    # Table 5 (NUMU)
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -','sys error +','sys error -'
    """

    arr_raw = [
        [3.0, 5.0, 0.792, 0.016, 0.016, 0.039, 0.039],
        [5.0, 7.0, 0.817, 0.013, 0.013, 0.035, 0.035],
        [7.0, 9.0, 0.779, 0.014, 0.014, 0.035, 0.035],
        [9.0, 11.0, 0.738, 0.016, 0.016, 0.026, 0.026],
        [11.0, 13.0, 0.717, 0.018, 0.018, 0.027, 0.027],
        [13.0, 17.0, 0.683, 0.015, 0.015, 0.029, 0.029],
        [17.0, 21.0, 0.654, 0.020, 0.020, 0.030, 0.030],
        [21.0, 25.0, 0.635, 0.024, 0.024, 0.041, 0.041],
        [25.0, 30.0, 0.609, 0.031, 0.031, 0.054, 0.054],
    ]

    arr_cs_cc_numu_ihep_jinr_energy_central = [(x[0] + x[1]) / 2.0 for x in arr_raw]
    arr_cs_cc_numu_ihep_jinr_energy_lo = [x[0] - (x[0] + x[1]) / 2.0 for x in arr_raw]
    arr_cs_cc_numu_ihep_jinr_energy_hi = [(x[0] + x[1]) / 2.0 - x[1] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_ihep_jinr_cs_central = [x[2] for x in arr_raw]
        arr_cs_cc_numu_ihep_jinr_cs_hi = [
            np.sqrt(x[3] ** 2.0 + x[5] ** 2.0) for x in arr_raw
        ]
        arr_cs_cc_numu_ihep_jinr_cs_lo = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_ihep_jinr_cs_central = [
            x[2] * (x[0] + x[1]) / 2.0 for x in arr_raw
        ]
        arr_cs_cc_numu_ihep_jinr_cs_hi = [
            np.sqrt(x[3] ** 2.0 + x[5] ** 2.0) * (x[0] + x[1]) / 2.0 for x in arr_raw
        ]
        arr_cs_cc_numu_ihep_jinr_cs_lo = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) * (x[0] + x[1]) / 2.0 for x in arr_raw
        ]

    """
    # Table 5 (NUMUBAR)
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -','sys error +','sys error -'
    """

    arr_raw = [
        [3.0, 5.0, 0.345, 0.016, 0.016, 0.023, 0.023],
        [5.0, 7.0, 0.347, 0.013, 0.013, 0.020, 0.020],
        [7.0, 10.0, 0.336, 0.012, 0.012, 0.019, 0.019],
        [10.0, 13.0, 0.346, 0.015, 0.015, 0.022, 0.022],
        [13.0, 17.0, 0.323, 0.017, 0.017, 0.022, 0.022],
        [17.0, 21.0, 0.284, 0.024, 0.024, 0.021, 0.021],
        [21.0, 26.0, 0.269, 0.029, 0.029, 0.023, 0.023],
        [26.0, 30.0, 0.279, 0.039, 0.039, 0.031, 0.031],
    ]

    arr_cs_cc_numubar_ihep_jinr_energy_central = [(x[0] + x[1]) / 2.0 for x in arr_raw]
    arr_cs_cc_numubar_ihep_jinr_energy_lo = [
        x[0] - (x[0] + x[1]) / 2.0 for x in arr_raw
    ]
    arr_cs_cc_numubar_ihep_jinr_energy_hi = [
        (x[0] + x[1]) / 2.0 - x[1] for x in arr_raw
    ]
    if divide_energy:
        arr_cs_cc_numubar_ihep_jinr_cs_central = [x[2] for x in arr_raw]
        arr_cs_cc_numubar_ihep_jinr_cs_hi = [
            np.sqrt(x[3] ** 2.0 + x[5] ** 2.0) for x in arr_raw
        ]
        arr_cs_cc_numubar_ihep_jinr_cs_lo = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) for x in arr_raw
        ]
    else:
        arr_cs_cc_numubar_ihep_jinr_cs_central = [
            x[2] * (x[0] + x[1]) / 2.0 for x in arr_raw
        ]
        arr_cs_cc_numubar_ihep_jinr_cs_hi = [
            np.sqrt(x[3] ** 2.0 + x[5] ** 2.0) * (x[0] + x[1]) / 2.0 for x in arr_raw
        ]
        arr_cs_cc_numubar_ihep_jinr_cs_lo = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) * (x[0] + x[1]) / 2.0 for x in arr_raw
        ]
    #################################################################################

    #################################################################################
    # CCFR
    # Seligman thesis 1997

    """
    # NUMU
    # 'E [GEV]','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [
        [76.06192263888305, 0.6993630573248406, 0.0, 0.0],
        [85.99780655502596, 0.7006369426751591, 0.0, 0.0],
        [95.92525414434554, 0.6968152866242037, 0.0, 0.0],
        [110.18897372084193, 0.7121019108280253, 0.0, 0.0],
        [130.04597798118698, 0.705732484076433, 0.0, 0.0],
        [150.91323237862235, 0.7095541401273884, 0.0, 0.0],
        [170.76812755726155, 0.7019108280254776, 0.0, 0.0],
        [190.93727591006876, 0.684076433121019, 0.0, 0.0],
        [216.079638925212, 0.6700636942675158, 0.0, 0.0],
        [244.89813135360868, 0.6764331210191081, 0.0, 0.0],
        [275.024254439617, 0.6726114649681527, 0.0, 0.0],
        [303.8469650314253, 0.681528662420382, 0.6993630573248406, 0.6649681528662419],
        [338.9484118614755, 0.6828025477707005, 0.7019108280254776, 0.6649681528662419],
    ]

    arr_cs_cc_numu_ccfr_energy_central = [x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_ccfr_cs_central = [x[1] for x in arr_raw]
        arr_cs_cc_numu_ccfr_cs_hi = [
            x[2] - x[1] if x[2] > x[1] else np.nan for x in arr_raw
        ]
        arr_cs_cc_numu_ccfr_cs_lo = [
            x[1] - x[3] if x[2] > x[1] else np.nan for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_ccfr_cs_central = [x[1] * x[0] for x in arr_raw]
        arr_cs_cc_numu_ccfr_cs_hi = [
            (x[2] - x[1]) * x[0] if x[2] > x[1] else np.nan for x in arr_raw
        ]
        arr_cs_cc_numu_ccfr_cs_lo = [
            (x[1] - x[3]) * x[0] if x[2] > x[1] else np.nan for x in arr_raw
        ]

    """
    # NUMUBAR
    # 'E [GEV]','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [
        [35.73952628138338, 0.34279666562548206, 0.0, 0.0],
        [45.65740076770575, 0.3350318471337579, 0.0, 0.0],
        [55.59539376555449, 0.3375796178343947, 0.0, 0.0],
        [65.52495043657993, 0.3350318471337579, 0.0, 0.0],
        [75.45872527101702, 0.3350318471337579, 0.0, 0.0],
        [85.39039102374828, 0.3337579617834393, 0.0, 0.0],
        [95.32627493989118, 0.3350318471337579, 0.0, 0.0],
        [109.56890369932931, 0.3375796178343947, 0.0, 0.0],
        [129.4617623486734, 0.3528662420382165, 0.0, 0.0],
        [150.31003501075634, 0.34522292993630566, 0.0, 0.0],
        [169.5068966971781, 0.34012738853503177, 0.0, 0.0],
        [190.36360568608427, 0.3375796178343947, 0.0, 0.0],
        [214.52946387143038, 0.3337579617834393, 0.0, 0.0],
        [243.3289745644746, 0.3286624203821654, 0.0, 0.0],
        [273.4303998332751, 0.3484849539767412, 0.3659053370524725, 0.3314200889229637],
        [
            304.42521440287425,
            0.31506625991309356,
            0.3371083772742228,
            0.29337966057391796,
        ],
        [
            334.5388857955451,
            0.36341671089879657,
            0.3996795491380738,
            0.32786490870342666,
        ],
    ]

    arr_cs_cc_numubar_ccfr_energy_central = [x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numubar_ccfr_cs_central = [x[1] for x in arr_raw]
        arr_cs_cc_numubar_ccfr_cs_hi = [
            x[2] - x[1] if x[2] > x[1] else np.nan for x in arr_raw
        ]
        arr_cs_cc_numubar_ccfr_cs_lo = [
            x[1] - x[3] if x[2] > x[1] else np.nan for x in arr_raw
        ]
    else:
        arr_cs_cc_numubar_ccfr_cs_central = [x[1] * x[0] for x in arr_raw]
        arr_cs_cc_numubar_ccfr_cs_hi = [
            (x[2] - x[1]) * x[0] if x[2] > x[1] else np.nan for x in arr_raw
        ]
        arr_cs_cc_numubar_ccfr_cs_lo = [
            (x[1] - x[3]) * x[0] if x[2] > x[1] else np.nan for x in arr_raw
        ]
    #################################################################################

    #################################################################################
    # CDHS
    # Z. Phys. C 35 (1987) 443

    """
    # NUMU
    # http://hepdata.cedar.ac.uk/view/ins246156/d1/plain.txt;jsessionid=sm1j32urlv2j
    # Path: /HepData/6594/d1-x1-y1
    # Measured charged current total cross section
    # RE : NUMU NUCLEON --> MU- X
    # SQRT(S) : 7.683 - 16.406 GeV
    # x : PLAB IN GEV
    # y : SIG/E IN 10**-38 CM**2/GEV
    # xdesc    x    xlow    xhigh    y    dy+    dy-    dy+    dy-
    #     50.0    10.0    100.0    0.691    +0.007    -0.007    +0.027    -0.027
    #     85.0    20.0    160.0    0.707    +0.002    -0.002    +0.022    -0.022
    #     112.0    20.0    200.0    0.708    +0.007    -0.007    +0.029    -0.029
    #     31.0    31.0    31.0    0.682    +0.008    -0.008    +0.02    -0.02
    #     50.0    50.0    50.0    0.706    +0.002    -0.002    +0.018    -0.018
    #     61.0    61.0    61.0    0.707    +0.009    -0.009    +0.023    -0.023
    #     83.0    83.0    83.0    0.724    +0.012    -0.012    +0.051    -0.051
    #     121.0    121.0    121.0    0.708    +0.003    -0.003    +0.023    -0.023
    #   143.0    143.0    143.0    0.711    +0.01    -0.01    +0.04    -0.04
    """

    arr_raw = [
        [50.0, 10.0, 100.0, 0.691, +0.007, -0.007, +0.027, -0.027],
        [85.0, 20.0, 160.0, 0.707, +0.002, -0.002, +0.022, -0.022],
        [112.0, 20.0, 200.0, 0.708, +0.007, -0.007, +0.029, -0.029],
        [31.0, 31.0, 31.0, 0.682, +0.008, -0.008, +0.02, -0.02],
        [50.0, 50.0, 50.0, 0.706, +0.002, -0.002, +0.018, -0.018],
        [61.0, 61.0, 61.0, 0.707, +0.009, -0.009, +0.023, -0.023],
        [83.0, 83.0, 83.0, 0.724, +0.012, -0.012, +0.051, -0.051],
        [121.0, 121.0, 121.0, 0.708, +0.003, -0.003, +0.023, -0.023],
        [143.0, 143.0, 143.0, 0.711, +0.01, -0.01, +0.04, -0.04],
    ]

    arr_cs_cc_numu_cdhs_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_cdhs_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numu_cdhs_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_cdhs_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_cdhs_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) for x in arr_raw
        ]
        arr_cs_cc_numu_cdhs_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_cdhs_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numu_cdhs_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) * x[0] for x in arr_raw
        ]
        arr_cs_cc_numu_cdhs_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) * x[0] for x in arr_raw
        ]

    """
    # NUMUBAR
    # http://hepdata.cedar.ac.uk/view/ins246156/d2/plain.txt;jsessionid=sm1j32urlv2j
    # Path: /HepData/6594/d2-x1-y1
    # Measured charged current total cross section
    # RE : NUMUBAR NUCLEON --> MU+ X
    # SQRT(S) : 7.683 - 16.406 GeV
    # x : PLAB IN GEV
    # y : SIG/E IN 10**-38 CM**2/GEV
    # xdesc    x    xlow    xhigh    y    dy+    dy-    dy+    dy-
    #     47.0    10.0    100.0    0.332    +0.004    -0.004    +0.012    -0.012
    #     72.0    20.0    160.0    0.333    +0.004    -0.004    +0.009    -0.009
    #     87.0    20.0    200.0    0.325    +0.007    -0.007    +0.012    -0.012
    #     31.0    31.0    31.0    0.327    +0.004    -0.004    +0.008    -0.008
    #     50.0    50.0    50.0    0.332    +0.005    -0.005    +0.007    -0.007
    #     61.0    61.0    61.0    0.338    +0.009    -0.009    +0.01    -0.01
    #     83.0    83.0    83.0    0.346    +0.007    -0.007    +0.022    -0.022
    #     121.0    121.0    121.0    0.337    +0.008    -0.008    +0.012    -0.012
    #     143.0    143.0    143.0    0.306    +0.015    -0.015    +0.02    -0.02
    """

    arr_raw = [
        [47.0, 10.0, 100.0, 0.332, +0.004, -0.004, +0.012, -0.012],
        [72.0, 20.0, 160.0, 0.333, +0.004, -0.004, +0.009, -0.009],
        [87.0, 20.0, 200.0, 0.325, +0.007, -0.007, +0.012, -0.012],
        [31.0, 31.0, 31.0, 0.327, +0.004, -0.004, +0.008, -0.008],
        [50.0, 50.0, 50.0, 0.332, +0.005, -0.005, +0.007, -0.007],
        [61.0, 61.0, 61.0, 0.338, +0.009, -0.009, +0.01, -0.01],
        [83.0, 83.0, 83.0, 0.346, +0.007, -0.007, +0.022, -0.022],
        [121.0, 121.0, 121.0, 0.337, +0.008, -0.008, +0.012, -0.012],
        [143.0, 143.0, 143.0, 0.306, +0.015, -0.015, +0.02, -0.02],
    ]

    arr_cs_cc_numubar_cdhs_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numubar_cdhs_energy_lo = [x[0] - x[1] for x in arr_raw]
    arr_cs_cc_numubar_cdhs_energy_hi = [x[2] - x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numubar_cdhs_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numubar_cdhs_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) for x in arr_raw
        ]
        arr_cs_cc_numubar_cdhs_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) for x in arr_raw
        ]
    else:
        arr_cs_cc_numubar_cdhs_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numubar_cdhs_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) * x[0] for x in arr_raw
        ]
        arr_cs_cc_numubar_cdhs_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) * x[0] for x in arr_raw
        ]

    #################################################################################

    #################################################################################
    # BNL
    # Phys. Rev. D 25 (1982) 617-623

    """
    # NUMU
    # http://hepdata.cedar.ac.uk/view/ins177607
    # *dataset:
    # *dscomment: Measured charged current total cross section.
    # Axis error includes +- 0.0/0.0 contribution (?////SYSTEMATIC ERROR NOT GIVENNEUTRAL CURRENT AND NEUTRAL PARTICLES INDUCED REACTIONS, RESCATTERING IN DEUTERIUM).
    # *reackey: NUMU NUCLEON --> MU- X
    # *obskey: SIG
    # *qual: RE : NUMU NUCLEON --> MU- X
    # *qual: SQRT(S) IN GEV : 1.481 TO 3.867
    # *yheader: SIG/E IN 10**-38 CM**2/GEV
    # *xheader: PLAB IN GEV
    # *data: x : y
    #  0.7; 1.09 +- 0.08;
    #  0.9; 0.97 +- 0.07;
    #  1.2; 0.98 +- 0.07;
    #  1.3; 0.85 +- 0.07;
    #  1.6; 0.78 +- 0.06;
    #  1.8; 0.79 +- 0.07;
    #  2.0; 0.8 +- 0.08;
    #  2.7; 0.83 +- 0.08;
    #  3.0; 0.79 +- 0.1;
    #  3.5; 0.91 +- 0.14;
    #  4.0; 0.85 +- 0.17;
    #  5.0; 0.78 +- 0.11;
    #  7.5; 0.7 +- 0.15;
    # *dataend:
    """

    arr_raw = [
        [0.7, 1.09, 0.08, 0.08],
        [0.9, 0.97, 0.07, 0.07],
        [1.2, 0.98, 0.07, 0.07],
        [1.3, 0.85, 0.07, 0.07],
        [1.6, 0.78, 0.06, 0.06],
        [1.8, 0.79, 0.07, 0.07],
        [2.0, 0.8, 0.08, 0.08],
        [2.7, 0.83, 0.08, 0.08],
        [3.0, 0.79, 0.1, 0.1],
        [3.5, 0.91, 0.14, 0.14],
        [4.0, 0.85, 0.17, 0.17],
        [5.0, 0.78, 0.11, 0.11],
        [7.5, 0.7, 0.15, 0.15],
    ]

    arr_cs_cc_numu_bnl_energy_central = [x[0] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_bnl_cs_central = [x[1] for x in arr_raw]
        arr_cs_cc_numu_bnl_cs_hi = [x[2] for x in arr_raw]
        arr_cs_cc_numu_bnl_cs_lo = [x[3] for x in arr_raw]
    else:
        arr_cs_cc_numu_bnl_cs_central = [x[1] * x[0] for x in arr_raw]
        arr_cs_cc_numu_bnl_cs_hi = [x[2] * x[0] for x in arr_raw]
        arr_cs_cc_numu_bnl_cs_lo = [x[3] * x[0] for x in arr_raw]
    # No NUMUBAR data
    #################################################################################

    #################################################################################
    # ArgoNeuT 2014
    # Phys. Rev. D 89, 112003 (2014)
    # https://arxiv.org/abs/1404.4809

    """
    # NUMU
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [[9.6, 6.5, 6.5, 0.66, 0.03, 0.03, 0.08, 0.08]]

    arr_cs_cc_numu_argoneut_14_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_argoneut_14_energy_lo = [x[1] for x in arr_raw]
    arr_cs_cc_numu_argoneut_14_energy_hi = [x[2] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_argoneut_14_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_argoneut_14_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) for x in arr_raw
        ]
        arr_cs_cc_numu_argoneut_14_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_argoneut_14_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numu_argoneut_14_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) * x[0] for x in arr_raw
        ]
        arr_cs_cc_numu_argoneut_14_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) * x[0] for x in arr_raw
        ]

    """
    # NUMUBAR
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [[3.6, 1.5, 1.5, 0.28, 0.01, 0.01, 0.03, 0.03]]

    arr_cs_cc_numubar_argoneut_14_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numubar_argoneut_14_energy_lo = [x[1] for x in arr_raw]
    arr_cs_cc_numubar_argoneut_14_energy_hi = [x[2] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numubar_argoneut_14_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numubar_argoneut_14_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) for x in arr_raw
        ]
        arr_cs_cc_numubar_argoneut_14_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) for x in arr_raw
        ]
    else:
        arr_cs_cc_numubar_argoneut_14_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numubar_argoneut_14_cs_hi = [
            np.sqrt(x[4] ** 2.0 + x[6] ** 2.0) * x[0] for x in arr_raw
        ]
        arr_cs_cc_numubar_argoneut_14_cs_lo = [
            np.sqrt(x[5] ** 2.0 + x[7] ** 2.0) * x[0] for x in arr_raw
        ]
    #################################################################################

    #################################################################################
    # ArgoNeuT 2012
    # PRL 108, 161802 (2012)
    # https://arxiv.org/abs/1111.0103

    """
    # NUMU
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [[4.3, 0.0, 0.0, 0.73, 0.12, 0.12]]

    arr_cs_cc_numu_argoneut_12_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_argoneut_12_energy_lo = [x[1] for x in arr_raw]
    arr_cs_cc_numu_argoneut_12_energy_hi = [x[2] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_argoneut_12_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_argoneut_12_cs_hi = [x[4] for x in arr_raw]
        arr_cs_cc_numu_argoneut_12_cs_lo = [x[5] for x in arr_raw]
    else:
        arr_cs_cc_numu_argoneut_12_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numu_argoneut_12_cs_hi = [x[4] * x[0] for x in arr_raw]
        arr_cs_cc_numu_argoneut_12_cs_lo = [x[5] * x[0] for x in arr_raw]
    # No NUMUBAR data
    #################################################################################

    #################################################################################
    # GGM-SPS
    # PLB 104, 235 (1981)

    """
    # NUMU
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [
        [
            17.425742574257427,
            15.24752475247525,
            19.60396039603961,
            0.5795918367346939,
            0.6653061224489796,
            0.5020408163265309,
        ],
        [
            22.37623762376238,
            19.60396039603961,
            24.75247524752475,
            0.5632653061224491,
            0.6244897959183673,
            0.47755102040816344,
        ],
        [
            27.128712871287128,
            24.95049504950495,
            29.306930693069305,
            0.6367346938775511,
            0.7081632653061225,
            0.5489795918367351,
        ],
        [
            34.65346534653466,
            30.099009900990097,
            39.80198019801979,
            0.7142857142857144,
            0.8081632653061226,
            0.616326530612245,
        ],
        [
            44.75247524752476,
            40.00,
            49.60396039603961,
            0.6367346938775511,
            0.7346938775510203,
            0.536734693877551,
        ],
        [
            59.60396039603961,
            50.29702970297029,
            69.30693069306932,
            0.6489795918367347,
            0.7428571428571429,
            0.5469387755102044,
        ],
        [
            84.35643564356435,
            69.9009900990099,
            99.10891089108912,
            0.6326530612244898,
            0.7183673469387755,
            0.5326530612244897,
        ],
        [
            119.40594059405939,
            99.50495049504948,
            139.40594059405942,
            0.526530612244898,
            0.6122448979591837,
            0.4346938775510203,
        ],
    ]

    arr_cs_cc_numu_ggm_sps_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_ggm_sps_energy_lo = [x[1] for x in arr_raw]
    arr_cs_cc_numu_ggm_sps_energy_hi = [x[2] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numu_ggm_sps_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_ggm_sps_cs_hi = [x[4] - x[3] for x in arr_raw]
        arr_cs_cc_numu_ggm_sps_cs_lo = [x[3] - x[5] for x in arr_raw]
    else:
        arr_cs_cc_numu_ggm_sps_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numu_ggm_sps_cs_hi = [(x[4] - x[3]) * x[0] for x in arr_raw]
        arr_cs_cc_numu_ggm_sps_cs_lo = [(x[3] - x[5]) * x[0] for x in arr_raw]

    """
    # NUMUBAR
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [
        [
            17.227722772277225,
            15.445544554455445,
            19.60396039603961,
            0.6257425742574321 / 2.0,
            0.6891089108910933 / 2.0,
            0.5465346534653568 / 2.0,
        ],
        [
            21.980198019801982,
            19.60396039603961,
            24.15841584158416,
            0.5702970297029792 / 2.0,
            0.637623762376244 / 2.0,
            0.49900990099011144 / 2.0,
        ],
        [
            27.524752475247524,
            24.95049504950495,
            29.306930693069305,
            0.5306930693069418 / 2.0,
            0.5900990099009986 / 2.0,
            0.4594059405940736 / 2.0,
        ],
        [
            34.45544554455445,
            29.900990099009903,
            39.20792079207921,
            0.6415841584158475 / 2.0,
            0.7306930693069327 / 2.0,
            0.5643564356435742 / 2.0,
        ],
        [
            44.75247524752474,
            40.00,
            49.801980198019805,
            0.5287128712871398 / 2.0,
            0.6000000000000076 / 2.0,
            0.4475247524752626 / 2.0,
        ],
        [
            59.999999999999986,
            50.0990099009901,
            69.50495049504951,
            0.6019801980198096 / 2.0,
            0.6950495049504983 / 2.0,
            0.5069306930693194 / 2.0,
        ],
        [
            84.85148514851485,
            69.9009900990099,
            99.60396039603958,
            0.629702970297036 / 2.0,
            0.7366336633663373 / 2.0,
            0.5069306930693189 / 2.0,
        ],
        [
            120.1980198019802,
            100.39603960396042,
            139.6039603960396,
            0.5702970297029792 / 2.0,
            0.6891089108910933 / 2.0,
            0.45742574257427204 / 2.0,
        ],
    ]

    arr_cs_cc_numubar_ggm_sps_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numubar_ggm_sps_energy_lo = [x[1] for x in arr_raw]
    arr_cs_cc_numubar_ggm_sps_energy_hi = [x[2] for x in arr_raw]
    if divide_energy:
        arr_cs_cc_numubar_ggm_sps_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numubar_ggm_sps_cs_hi = [x[4] - x[3] for x in arr_raw]
        arr_cs_cc_numubar_ggm_sps_cs_lo = [x[3] - x[5] for x in arr_raw]
    else:
        arr_cs_cc_numubar_ggm_sps_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numubar_ggm_sps_cs_hi = [(x[4] - x[3]) * x[0] for x in arr_raw]
        arr_cs_cc_numubar_ggm_sps_cs_lo = [(x[3] - x[5]) * x[0] for x in arr_raw]
    #################################################################################

    #################################################################################
    # NuTeV
    # PRD 74, 012008 (2006)
    # https://arxiv.org/abs/hep-ex/0509010

    """
    # NUMU
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [
        [36.374169971923884, 0.0, 0.0, 0.7030304380765633, 0.0, 0.0],
        [46.32002317393823, 0.0, 0.0, 0.6851619947412988, 0.0, 0.0],
        [56.3093275101386, 0.0, 0.0, 0.6816324256874193, 0.0, 0.0],
        [66.4534961451045, 0.0, 0.0, 0.6792080752261689, 0.0, 0.0],
        [76.46396898257497, 0.0, 0.0, 0.6826641115914258, 0.0, 0.0],
        [86.47221355675389, 0.0, 0.0, 0.6853848210704578, 0.0, 0.0],
        [96.45483310307947, 0.0, 0.0, 0.6796492713579038, 0.0, 0.0],
        [111.27724051873973, 0.0, 0.0, 0.6710437185257812, 0.0, 0.0],
        [
            131.38286019876108,
            0.0,
            0.0,
            0.6558982129328401,
            0.6617808280226392,
            0.6496479343999287,
        ],
        [
            151.40380587370205,
            0.0,
            0.0,
            0.662810285663354,
            0.6690605641962655,
            0.656927670573555,
        ],
        [171.3311644903962, 0.0, 0.0, 0.638838629172423, 0.0, 0.0],
        [
            191.57939302107934,
            0.0,
            0.0,
            0.6707540442978742,
            0.677737421453719,
            0.6648692009447836,
        ],
        [
            216.39333303623152,
            214.1205044788092,
            218.66616159365364,
            0.659354249298097,
            0.0,
            0.0,
        ],
        [
            246.48714292080754,
            244.06279245955702,
            248.91260751370353,
            0.6403115112081643,
            0.647294888364009,
            0.6336913409688488,
        ],
        [
            276.6154908864008,
            273.88809661749633,
            279.34177102366414,
            0.6826663398547169,
            0.6951691251838317,
            0.6716386648246355,
        ],
        [
            306.4251972013013,
            303.6966888007487,
            309.1514773385623,
            0.6198694237711126,
            0.6345759614956101,
            0.6047952226035026,
        ],
        [
            341.5927626008289,
            337.6509648380052,
            345.22706002941305,
            0.6251660056152233,
            0.6453874949864074,
            0.6038415259147019,
        ],
    ]

    arr_cs_cc_numu_nutev_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_nutev_energy_lo = [
        x[0] - x[1] if x[2] > x[0] else np.nan for x in arr_raw
    ]
    arr_cs_cc_numu_nutev_energy_hi = [
        x[2] - x[0] if x[2] > x[0] else np.nan for x in arr_raw
    ]
    if divide_energy:
        arr_cs_cc_numu_nutev_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_nutev_cs_hi = [
            x[4] - x[3] if x[4] > x[3] else np.nan for x in arr_raw
        ]
        arr_cs_cc_numu_nutev_cs_lo = [
            x[3] - x[5] if x[4] > x[3] else np.nan for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_nutev_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numu_nutev_cs_hi = [
            (x[4] - x[3]) * x[0] if x[4] > x[3] else np.nan for x in arr_raw
        ]
        arr_cs_cc_numu_nutev_cs_lo = [
            (x[3] - x[5]) * x[0] if x[4] > x[3] else np.nan for x in arr_raw
        ]

    """
    # NUMUBAR
    # 'E [GEV] LOW','E [GEV] HIGH','SIG/E [10**-38CM**2/GEV]','error +','error -'
    """

    arr_raw = [
        [35.156424083069666, 0.0, 0.0, 0.3511742947546682, 0.0, 0.0],
        [45.279424216765456, 0.0, 0.0, 0.34176433887428137, 0.0, 0.0],
        [55.42804937831454, 0.0, 0.0, 0.3408106421854807, 0.0, 0.0],
        [65.29368510183163, 0.0, 0.0, 0.3464704309461206, 0.0, 0.0],
        [75.43451134186014, 0.0, 0.0, 0.34294309015553276, 0.0, 0.0],
        [85.4327287312269, 0.0, 0.0, 0.34235482864655287, 0.0, 0.0],
        [95.58358215606755, 0.0, 0.0, 0.3421364588439769, 0.0, 0.0],
        [110.43384286287268, 0.0, 0.0, 0.3427224920896652, 0.0, 0.0],
        [130.26650028967427, 0.0, 0.0, 0.337499442934177, 0.0, 0.0],
        [150.41445697223583, 0.0, 0.0, 0.3363251481795087, 0.0, 0.0],
        [170.42871785730202, 0.0, 0.0, 0.3410312402513481, 0.0, 0.0],
        [190.5555060385936, 0.0, 0.0, 0.3328713400775434, 0.0, 0.0],
        [
            215.60229956771693,
            213.17572084317487,
            217.87401399349346,
            0.34831320468826577,
            0.0,
            0.0,
        ],
        [
            245.5423592851732,
            243.11912295556843,
            247.96782387806942,
            0.32853291144881647,
            0.3355185168679528,
            0.32191496947279263,
        ],
        [
            275.5247560051696,
            272.79513347297114,
            278.24992201078476,
            0.32272382904763997,
            0.3330206337180798,
            0.31243148090378337,
        ],
        [
            305.54169080618567,
            302.81429653727884,
            308.26908507509245,
            0.32831231338294903,
            0.34448950487989627,
            0.31139979499977677,
        ],
        [
            340.6747181246936,
            336.7351486251615,
            344.4627657203975,
            0.3222113284905741,
            0.3468447791791074,
            0.29831543295155727,
        ],
    ]

    arr_cs_cc_numubar_nutev_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numubar_nutev_energy_lo = [
        x[0] - x[1] if x[2] > x[1] else np.nan for x in arr_raw
    ]
    arr_cs_cc_numubar_nutev_energy_hi = [
        x[2] - x[0] if x[2] > x[1] else np.nan for x in arr_raw
    ]
    if divide_energy:
        arr_cs_cc_numubar_nutev_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numubar_nutev_cs_hi = [
            x[4] - x[3] if x[4] > x[3] else np.nan for x in arr_raw
        ]
        arr_cs_cc_numubar_nutev_cs_lo = [
            x[3] - x[5] if x[4] > x[3] else np.nan for x in arr_raw
        ]
    else:
        arr_cs_cc_numubar_nutev_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numubar_nutev_cs_hi = [
            (x[4] - x[3]) * x[0] if x[4] > x[3] else np.nan for x in arr_raw
        ]
        arr_cs_cc_numubar_nutev_cs_lo = [
            (x[3] - x[5]) * x[0] if x[4] > x[3] else np.nan for x in arr_raw
        ]

    #################################################################################

    #################################################################################
    # FASERnu
    # PRL 133, 021802 (2024)
    # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.133.021802

    arr_raw = [[930, 520, 1760, 0.5, 0.7, 0.3]]
    arr_cs_cc_numu_fasernu_energy_central = [x[0] for x in arr_raw]
    arr_cs_cc_numu_fasernu_energy_lo = [
        x[0] - x[1] if x[2] > x[0] else np.nan for x in arr_raw
    ]
    arr_cs_cc_numu_fasernu_energy_hi = [
        x[2] - x[0] if x[2] > x[0] else np.nan for x in arr_raw
    ]
    if divide_energy:
        arr_cs_cc_numu_fasernu_cs_central = [x[3] for x in arr_raw]
        arr_cs_cc_numu_fasernu_cs_hi = [
            x[4] - x[3] if x[4] > x[3] else np.nan for x in arr_raw
        ]
        arr_cs_cc_numu_fasernu_cs_lo = [
            x[3] - x[5] if x[4] > x[3] else np.nan for x in arr_raw
        ]
    else:
        arr_cs_cc_numu_fasernu_cs_central = [x[3] * x[0] for x in arr_raw]
        arr_cs_cc_numu_fasernu_cs_hi = [
            (x[4] - x[3]) * x[0] if x[4] > x[3] else np.nan for x in arr_raw
        ]
        arr_cs_cc_numu_fasernu_cs_lo = [
            (x[3] - x[5]) * x[0] if x[4] > x[3] else np.nan for x in arr_raw
        ]
    #################################################################################

    #################################################################################
    ### ICECUBE

    icecube17_band = np.array(
        [
            [5973.429965216526, 2213.7712517843042, 4610.168661060654],
            [10665.274261466224, 3465.9557108295326, 7217.837157209308],
            [23101.29700083158, 6132.119639329999, 12770.111529964062],
            [48452.33210350108, 9999.99999999996, 20824.954960205752],
            [135789.79506128453, 18428.36161290151, 39973.291581617945],
            [313692.0385392787, 30052.188634263133, 62583.5474764137],
            [637082.6886069496, 43367.874909459824, 90313.4041709338],
            [968309.0391711551, 53169.63963307794, 110725.5350609221],
        ]
    )
    if divide_energy:
        icecube17_band[:, 1] = icecube17_band[:, 1] / icecube17_band[:, 0]
        icecube17_band[:, 2] = icecube17_band[:, 2] / icecube17_band[:, 0]

    #################################################################################

    ################################################################################
    # IceCube 6-yr HESE - 4 bins -- including self-veto
    # Our work, diff-nu-nubar-sigma

    """
    # Average between NUMU and NUMUBAR
    # 'E [GEV] LOW','E [GEV] HIGH','SIG [CM**2]','error +','error -'
    """

    # Hi-res, obtained on ruby
    arr_raw = [
        [
            18.0e3,
            50.0e3,
            10.0 ** (-34.35),
            10.0 ** (-34.35 + 0.53),
            10.0 ** (-34.35 - 0.53),
        ],
        [
            50.0e3,
            100.0e3,
            10.0 ** (-33.80),
            10.0 ** (-33.80 + 0.67),
            10.0 ** (-33.80 - 0.67),
        ],
        [
            100.0e3,
            400.0e3,
            10.0 ** (-33.84),
            10.0 ** (-33.84 + 0.67),
            10.0 ** (-33.84 - 0.67),
        ],
        [
            400.0e3,
            2004.0e3,
            10.0 ** (-31.71),
            10.0 ** (-31.71 + 1.50),
            10.0 ** (-31.71 - 1.50),
        ],
    ]

    arr_cs_cc_nuavg_ic_energy_central = [(x[0] + x[1]) / 2.0 for x in arr_raw]
    arr_cs_cc_nuavg_ic_energy_lo = [(x[0] + x[1]) / 2.0 - x[0] for x in arr_raw]
    arr_cs_cc_nuavg_ic_energy_hi = [x[1] - (x[0] + x[1]) / 2.0 for x in arr_raw]
    if divide_energy:
        arr_cs_cc_nuavg_ic_cs_central = [
            1.0e38 * x[2] / ((x[0] + x[1]) / 2.0) for x in arr_raw
        ]
        arr_cs_cc_nuavg_ic_cs_hi = [
            1.0e38 * (x[3] - x[2]) / ((x[0] + x[1]) / 2.0) for x in arr_raw
        ]
        arr_cs_cc_nuavg_ic_cs_lo = [
            1.0e38 * (x[2] - x[4]) / ((x[0] + x[1]) / 2.0) for x in arr_raw
        ]
    else:
        arr_cs_cc_nuavg_ic_cs_central = [1.0e38 * x[2] for x in arr_raw]
        arr_cs_cc_nuavg_ic_cs_hi = [1.0e38 * (x[3] - x[2]) for x in arr_raw]
        arr_cs_cc_nuavg_ic_cs_lo = [1.0e38 * (x[2] - x[4]) for x in arr_raw]
    ################################################################################

    ################################################################################
    # IceCube 7.5-yr HESE -
    # https://arxiv.org/abs/2011.03560

    icecube20 = [
        {
            "center": [77315.78263519185, 3610.124038653262],
            "bot": [77315.78263519185, 2040.4901998846856],
            "top": [74868.93854837948, 11300.49181671491],
            "left": [59777.30244492257, 2944.601633495886],
            "right": [100000, 4079.6177156127046],
        },
        {
            "center": [133566.7393432868, 38376.98005790602],
            "bot": [133566.7393432868, 18428.36161290155],
            "top": [133566.7393432868, 67898.22281224042],
            "left": [96835.25924020249, 31302.2258950705],
            "right": [202890.90056259086, 47050.73061902763],
        },
        {
            "center": [279851.3293938475, 24512.10063563054],
            "bot": [279851.3293938475, 8849.172374257772],
            "top": [279851.3293938475, 60084.307757127914],
            "left": [196469.92953456633, 19194.899416590662],
            "right": [515573.40012740105, 32604.25914100972],
        },
        {
            "center": [808762.9152645408, 271411.545834736],
            "bot": [808762.9152645408, 62583.54747641383],
            "top": [808762.9152645408, 960065.5472554038],
            "left": [483456.65392005735, 195900.4340421579],
            "right": [9935888.37828336, 815651.0972942476],
        },
    ]

    #################################################################################
    # DIS line (full calculation)
    arr_dis_energy = [
        0.01,
        0.012328467394420659,
        0.015199110829529339,
        0.018738174228603841,
        0.023101297000831605,
        0.028480358684358019,
        0.035111917342151307,
        0.043287612810830572,
        0.0533669923120631,
        0.065793322465756823,
        0.081113083078968723,
        0.10000000000000001,
        0.12328467394420659,
        0.15199110829529339,
        0.18738174228603841,
        0.23101297000831603,
        0.2848035868435802,
        0.35111917342151311,
        0.43287612810830595,
        0.533669923120631,
        0.65793322465756821,
        0.81113083078968728,
        1.0,
        1.2328467394420659,
        1.5199110829529332,
        1.873817422860385,
        2.3101297000831602,
        2.8480358684358018,
        3.5111917342151311,
        4.3287612810830574,
        5.3366992312063131,
        6.5793322465756825,
        8.1113083078968717,
        10.0,
        12.32846739442066,
        15.199110829529348,
        18.73817422860385,
        23.101297000831604,
        28.48035868435802,
        35.111917342151308,
        43.287612810830616,
        53.366992312063125,
        65.79332246575683,
        81.113083078968728,
        100.0,
        123.28467394420659,
        151.99110829529332,
        187.3817422860383,
        231.01297000831579,
        284.80358684358049,
        351.11917342151344,
        432.87612810830615,
        533.66992312063121,
        657.93322465756819,
        811.13083078968725,
        1000.0,
        1232.8467394420659,
        1519.9110829529332,
        1873.817422860383,
        2310.1297000831628,
        2848.0358684358048,
        3511.1917342151346,
        4328.7612810830615,
        5336.699231206313,
        6579.3322465756828,
        8111.3083078968721,
        10000.0,
        12328.467394420659,
        15199.110829529331,
        18738.174228603832,
        23101.297000831626,
        28480.358684358049,
        35111.917342151348,
        43287.612810830615,
        53366.992312063128,
        65793.322465756821,
        81113.083078968717,
        100000.0,
        123284.67394420659,
        151991.10829529332,
        187381.74228603867,
        231012.97000831628,
        284803.58684358047,
        351119.17342151346,
        432876.12810830615,
        533669.92312063125,
        657933.22465756827,
        811130.83078968723,
        1000000.0,
        1232846.7394420684,
        1519911.0829529332,
        1873817.4228603868,
        2310129.700083158,
        2848035.8684358047,
        3511191.7342151273,
        4328761.2810830614,
        5336699.2312063016,
        6579332.2465756824,
        8111308.3078968888,
        10000000.0,
    ]  # [GeV]
    arr_dis_cs_raw = [
        3.9146146595649999e-37,
        3.1906829797467376e-37,
        2.6034796683606003e-37,
        2.1271809505327561e-37,
        1.7408403659533781e-37,
        1.4274675944282592e-37,
        1.1732812628873826e-37,
        9.6710287408230948e-38,
        7.9986520343624425e-38,
        6.6421354454231947e-38,
        5.5418226704119274e-38,
        4.6493246269650335e-38,
        3.9253914196547486e-38,
        3.3381862249764967e-38,
        2.8618676171441036e-38,
        2.4751494585265832e-38,
        2.1596794867858922e-38,
        1.8999376590565093e-38,
        1.6841797050259346e-38,
        1.5037139966395841e-38,
        1.352049006634688e-38,
        1.2242568022959765e-38,
        1.1166638957990549e-38,
        1.0260347696355845e-38,
        9.4947788121194603e-39,
        8.8482957604583784e-39,
        8.3019307547061459e-39,
        7.8394611601493065e-39,
        7.4477312241119139e-39,
        7.1147416417711783e-39,
        6.829776416483251e-39,
        6.5899901542038582e-39,
        6.3863276747282906e-39,
        6.2103953549528084e-39,
        6.0597141824432318e-39,
        5.9308560921854949e-39,
        5.8196213023870665e-39,
        5.7243577413567575e-39,
        5.6396491467004515e-39,
        5.5655877780223423e-39,
        5.5013823742379428e-39,
        5.4447810649852793e-39,
        5.3941244978134042e-39,
        5.3480094251799645e-39,
        5.3058749244755088e-39,
        5.2656042172763661e-39,
        5.2295640164902756e-39,
        5.1948058874061e-39,
        5.161662376476451e-39,
        5.1292258435352807e-39,
        5.0963766155801884e-39,
        5.0632762725742924e-39,
        5.0285774462847825e-39,
        4.990827944314087e-39,
        4.9502074480457819e-39,
        4.9051839504853769e-39,
        4.8548288265854508e-39,
        4.7987460651454269e-39,
        4.7356405245702886e-39,
        4.6640006715605618e-39,
        4.5828965669436114e-39,
        4.491392600588347e-39,
        4.3886793018736097e-39,
        4.2754258372420212e-39,
        4.1496470206296528e-39,
        4.0127270744319147e-39,
        3.865228641804291e-39,
        3.7081277878136196e-39,
        3.5428258076465466e-39,
        3.3706297729674556e-39,
        3.1927363445945573e-39,
        3.0112344464795341e-39,
        2.8281442010715574e-39,
        2.6462902599409864e-39,
        2.4674392089724955e-39,
        2.2923996637739577e-39,
        2.1224337035251038e-39,
        1.9585813760504172e-39,
        1.8018021614928074e-39,
        1.6526534084105611e-39,
        1.5116647989100753e-39,
        1.379065946196128e-39,
        1.254925926257109e-39,
        1.139220521117625e-39,
        1.0318351952295794e-39,
        9.3250009423533595e-40,
        8.4086229607135346e-40,
        7.5656016531790672e-40,
        6.7917639646467664e-40,
        6.0823935419015025e-40,
        5.4322249577114891e-40,
        4.8361149226985023e-40,
        4.2902302294424997e-40,
        3.7907610791368109e-40,
        3.3343938131109214e-40,
        2.9186004033953091e-40,
        2.5417013450426419e-40,
        2.2017812039912533e-40,
        1.8966629934023095e-40,
        1.6246463567699999e-40,
    ]  # [cm^2 GeV^-1]
    avg_sigma_cc_low_energy = (0.33 + 0.675) / 2.0

    # renorm_const = avg_sigma_cc_low_energy/(arr_dis_cs_raw[44]*1.e38)
    renorm_const = 1.0
    # print arr_dis_energy[44]
    arr_dis_cs_renorm = [x * renorm_const * 1.0e38 for x in arr_dis_cs_raw]
    # print arr_dis_cs_renorm
    if divide_energy == False:
        arr_dis_cs_renorm = [xs * e for xs, e in zip(arr_dis_cs_renorm, arr_dis_energy)]
    if print_xsref:
        ax.plot(
            arr_dis_energy[44 : len(arr_dis_energy)],
            arr_dis_cs_renorm[44 : len(arr_dis_cs_renorm)],
            color="k",
            ls="dotted",
            lw=1,
        )

    #################################################################################

    col_lines = []
    lhc_lines = []
    cosmic_lines = []
    if experiments == None:
        experiments = [
            "T2K",
            "Argoneut",
            "ANL",
            "BEBC",
            "BNL",
            "CCFR",
            "CDHSW",
            "GGM",
            "IHEP",
            "MINOS",
            "NOMAD",
            "NuTeV",
            "SciBoone",
            "SKAT",
            "IceCube",
            "IceCube-HESE",
            "IceCube-HESE-20",
        ]
        colors = ["k" for _ in experiments]

    # Cosmic Neutrino Cross Section Measurements

    if "IceCube" in experiments:
        col_line = ax.fill_between(
            icecube17_band[:, 0],
            icecube17_band[:, 1],
            icecube17_band[:, 2],
            color="silver",
            label=r"IceCube 17",
        )
        cosmic_lines.append([col_line])

    if "IceCube-HESE" in experiments:
        ax.errorbar(
            arr_cs_cc_nuavg_ic_energy_central[0:3],
            arr_cs_cc_nuavg_ic_cs_central[0:3],
            xerr=[arr_cs_cc_nuavg_ic_energy_lo[0:3], arr_cs_cc_nuavg_ic_energy_hi[0:3]],
            yerr=[arr_cs_cc_nuavg_ic_cs_lo[0:3], arr_cs_cc_nuavg_ic_cs_hi[0:3]],
            linestyle="None",
            color="k",
            markeredgecolor="k",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="",
            marker=r"$\ast$",
        )  # , label='IceCube'
        ax.errorbar(
            [arr_cs_cc_nuavg_ic_energy_central[3]],
            [arr_cs_cc_nuavg_ic_cs_central[3] - arr_cs_cc_nuavg_ic_cs_lo[3]],
            xerr=[[arr_cs_cc_nuavg_ic_energy_lo[3]], [arr_cs_cc_nuavg_ic_energy_hi[3]]],
            linestyle="None",
            color="k",
            markeredgecolor="k",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="",
            marker=None,
        )  # , label='IceCube'
        ax.plot(
            [
                arr_cs_cc_nuavg_ic_energy_central[3],
                arr_cs_cc_nuavg_ic_energy_central[3],
            ],
            [
                arr_cs_cc_nuavg_ic_cs_central[3] - arr_cs_cc_nuavg_ic_cs_lo[3],
                0.168 * arr_cs_cc_nuavg_ic_cs_central[3],
            ],
            "k-",
        )
        ax.annotate(
            r"$>$",
            xy=(
                1.025 * arr_cs_cc_nuavg_ic_energy_central[3],
                0.151 * arr_cs_cc_nuavg_ic_cs_central[3],
            ),
            xycoords="data",
            color="k",
            fontsize=20,
            horizontalalignment="center",
            rotation=90,
        )
        cosmic_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            markeredgecolor="k",
            markersize=markersize,
            marker=r"$\ast$",
            label=r"IC HESE showers 17",
        )
        cosmic_lines.append(cosmic_line)

    if "IceCube-HESE-20" in experiments:
        for ip, p in enumerate(icecube20):
            x, y = p["center"]
            x1, y1 = p["left"]
            x2, y2 = p["right"]
            x3, y3 = p["top"]
            x4, y4 = p["bot"]
            if ip == 0:
                lab = r"IceCube HESE 20"
                if divide_energy:
                    cosmic_line = ax.plot(
                        x,
                        y / x,
                        color="k",
                        markersize=markersize,
                        marker=r"$\bigcirc$",
                        label=lab,
                    )
                else:
                    cosmic_line = ax.plot(
                        x,
                        y,
                        color="k",
                        markersize=markersize,
                        marker=r"$\bigcirc$",
                        label=lab,
                    )
                cosmic_lines.append(cosmic_line)
            else:
                if divide_energy:
                    ax.plot(
                        x, y / x, color="k", markersize=markersize, marker=r"$\bigcirc$"
                    )
                else:
                    ax.plot(
                        x, y, color="k", markersize=markersize, marker=r"$\bigcirc$"
                    )
            if divide_energy:
                ax.plot([x1, x2], [y1 / x1, y2 / x2], "k-", linewidth=elinewidth)
            else:
                ax.plot([x1, x2], [y1, y2], "k-", linewidth=elinewidth)
            if divide_energy:
                ax.plot([x3, x4], [y3 / x3, y4 / x4], "k-", linewidth=elinewidth)
            else:
                ax.plot([x3, x4], [y3, y4], "k-", linewidth=elinewidth)

    # NUMU Cross Section Measurements

    if "MINOS" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_minos_energy_central,
            arr_cs_cc_numu_minos_cs_central,
            yerr=[arr_cs_cc_numu_minos_cs_lo, arr_cs_cc_numu_minos_cs_hi],
            linestyle="None",
            color="k",
            markersize=int(markersize * 0.8),
            elinewidth=elinewidth,
            fmt="o",
            zorder=10,
        )  # , label='MINOS'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            markersize=int(markersize * 0.8),
            marker=r"o",
            label="MINOS 10",
        )
        col_lines.append(col_line)

    if "NOMAD" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_nomad_energy_central,
            arr_cs_cc_numu_nomad_cs_central,
            xerr=[arr_cs_cc_numu_nomad_energy_lo, arr_cs_cc_numu_nomad_energy_hi],
            yerr=[arr_cs_cc_numu_nomad_cs_lo, arr_cs_cc_numu_nomad_cs_hi],
            linestyle="None",
            color="0.5",
            markeredgecolor="0.5",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="^",
        )  # , label='NOMAD'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="0.5",
            markeredgecolor="0.5",
            markersize=markersize,
            marker=r"^",
            label="NOMAD 08",
        )
        col_lines.append(col_line)
        # print ('NOMAD nu', arr_cs_cc_numu_nomad_energy_central, arr_cs_cc_numu_nomad_cs_central, [arr_cs_cc_numu_nomad_cs_lo, arr_cs_cc_numu_nomad_cs_hi])

    if "T2K" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_t2k_fe_2014_energy_central,
            arr_cs_cc_numu_t2k_fe_2014_cs_central,
            xerr=[
                arr_cs_cc_numu_t2k_fe_2014_energy_lo,
                arr_cs_cc_numu_t2k_fe_2014_energy_hi,
            ],
            yerr=[arr_cs_cc_numu_t2k_fe_2014_cs_lo, arr_cs_cc_numu_t2k_fe_2014_cs_hi],
            linestyle="None",
            marker="+",
            color="k",
            mew=1,
            markersize=int(1.3 * markersize),
            elinewidth=elinewidth,
        )  # , label='T2K (Fe)'
        ax.plot(
            arr_cs_cc_numu_t2k_fe_2014_energy_central,
            arr_cs_cc_numu_t2k_fe_2014_cs_central,
            linestyle="None",
            marker="+",
            color="k",
            mew=3,
            markersize=int(1.3 * markersize),
        )  # , label='T2K (Fe)'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            marker="+",
            color="k",
            mew=3,
            markersize=int(1.3 * markersize),
            label="T2K (Fe) 14",
        )
        col_lines.append(col_line)

        ax.errorbar(
            arr_cs_cc_numu_t2k_ch_2014_energy_central,
            arr_cs_cc_numu_t2k_ch_2014_cs_central,
            xerr=[
                arr_cs_cc_numu_t2k_ch_2014_energy_lo,
                arr_cs_cc_numu_t2k_ch_2014_energy_hi,
            ],
            yerr=[arr_cs_cc_numu_t2k_ch_2014_cs_lo, arr_cs_cc_numu_t2k_ch_2014_cs_hi],
            linestyle="None",
            marker="+",
            color="0.5",
            mew=1,
            markersize=int(1.3 * markersize),
            elinewidth=elinewidth,
        )  # , label='T2K (CH)'
        ax.plot(
            arr_cs_cc_numu_t2k_ch_2014_energy_central,
            arr_cs_cc_numu_t2k_ch_2014_cs_central,
            linestyle="None",
            marker="+",
            color="0.5",
            mew=3,
            markersize=int(1.3 * markersize),
        )  # , label='T2K (Fe)'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            marker="+",
            color="0.5",
            mew=3,
            markersize=int(1.3 * markersize),
            label="T2K (CH) 14",
        )
        col_lines.append(col_line)

        ax.errorbar(
            arr_cs_cc_numu_t2k_c_2013_energy_central,
            arr_cs_cc_numu_t2k_c_2013_cs_central,
            xerr=[
                arr_cs_cc_numu_t2k_c_2013_energy_lo,
                arr_cs_cc_numu_t2k_c_2013_energy_hi,
            ],
            yerr=[arr_cs_cc_numu_t2k_c_2013_cs_lo, arr_cs_cc_numu_t2k_c_2013_cs_hi],
            linestyle="None",
            marker="*",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=int(1.3 * markersize),
            elinewidth=elinewidth,
        )  # , label='T2K (C)'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            marker="*",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=int(1.3 * markersize),
            label="T2K (C) 13",
        )
        col_lines.append(col_line)

    if "Argoneut" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_argoneut_14_energy_central,
            arr_cs_cc_numu_argoneut_14_cs_central,
            xerr=[
                arr_cs_cc_numu_argoneut_14_energy_lo,
                arr_cs_cc_numu_argoneut_14_energy_hi,
            ],
            yerr=[arr_cs_cc_numu_argoneut_14_cs_lo, arr_cs_cc_numu_argoneut_14_cs_hi],
            linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="^",
        )  # , label='ArgoNeuT 2014'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=markersize,
            marker=r"^",
            label="ArgoNeuT 14",
        )
        col_lines.append(col_line)

        ax.errorbar(
            arr_cs_cc_numu_argoneut_12_energy_central,
            arr_cs_cc_numu_argoneut_12_cs_central,
            yerr=[arr_cs_cc_numu_argoneut_12_cs_lo, arr_cs_cc_numu_argoneut_12_cs_hi],
            linestyle="None",
            color="0.5",
            markeredgecolor="0.5",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="o",
            zorder=10,
        )  # , label='ArgoNeuT 2012'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="0.5",
            markeredgecolor="0.5",
            markersize=int(0.8 * markersize),
            marker=r"o",
            label="ArgoNeuT 12",
        )
        col_lines.append(col_line)

    if "ANL" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_anl_energy_central,
            arr_cs_cc_numu_anl_cs_central,
            yerr=[arr_cs_cc_numu_anl_cs_lo, arr_cs_cc_numu_anl_cs_hi],
            linestyle="None",
            color="k",
            markersize=int(1.3 * markersize),
            elinewidth=elinewidth,
            fmt="*",
        )  # , label='ANL'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            marker="*",
            color="k",
            markersize=int(1.3 * markersize),
            label="ANL 79",
        )
        col_lines.append(col_line)

    if "BEBC" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_bebc_2_energy_central,
            arr_cs_cc_numu_bebc_2_cs_central,
            yerr=[arr_cs_cc_numu_bebc_2_cs_lo, arr_cs_cc_numu_bebc_2_cs_hi],
            linestyle="None",
            color="k",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="",
            marker=r"$\bigcirc$",
        )  # , label='BEBC'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            markersize=markersize,
            marker=r"$\bigcirc$",
            label="BEBC 79",
        )
        col_lines.append(col_line)

    if "BNL" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_bnl_energy_central,
            arr_cs_cc_numu_bnl_cs_central,
            yerr=[arr_cs_cc_numu_bnl_cs_lo, arr_cs_cc_numu_bnl_cs_hi],
            linestyle="None",
            color="k",
            markeredgecolor="k",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="^",
        )  # , label='BNL'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            markeredgecolor="k",
            markersize=markersize,
            marker=r"^",
            label="BNL 82",
        )
        col_lines.append(col_line)

    if "CDHSW" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_cdhs_energy_central,
            arr_cs_cc_numu_cdhs_cs_central,
            yerr=[arr_cs_cc_numu_cdhs_cs_lo, arr_cs_cc_numu_cdhs_cs_hi],
            linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="s",
        )  # , label='CDHS'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=int(0.8 * markersize),
            marker=r"s",
            label="CDHS 87",
        )
        col_lines.append(col_line)
        # print ('CDHSW nu', arr_cs_cc_numu_cdhs_energy_central, arr_cs_cc_numu_cdhs_cs_central, [arr_cs_cc_numu_cdhs_cs_lo, arr_cs_cc_numu_cdhs_cs_hi])

    if "CCFR" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_ccfr_energy_central,
            arr_cs_cc_numu_ccfr_cs_central,
            yerr=[arr_cs_cc_numu_ccfr_cs_lo, arr_cs_cc_numu_ccfr_cs_hi],
            linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="",
            marker="d",
            zorder=2,
        )  # , label='CCFR'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=int(0.8 * markersize),
            marker="d",
            label="CCFR 97",
        )
        col_lines.append(col_line)
        # print ('CCFR nu', arr_cs_cc_numu_ccfr_energy_central, arr_cs_cc_numu_ccfr_cs_central, [arr_cs_cc_numu_ccfr_cs_lo, arr_cs_cc_numu_ccfr_cs_hi])

    if "GGM" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_ggm_sps_energy_central,
            arr_cs_cc_numu_ggm_sps_cs_central,
            yerr=[arr_cs_cc_numu_ggm_sps_cs_lo, arr_cs_cc_numu_ggm_sps_cs_hi],
            # linestyle="None",
            color="0.5",
            markeredgecolor="0.5",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            marker="s",
            linestyle="",
            # fmt="s",
            # fmt="--s",
        )  # , label='GGM-SPS'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="0.5",
            markeredgecolor="0.5",
            markersize=int(0.8 * markersize),
            marker=r"s",
            label="GGM-SPS 81",
        )
        col_lines.append(col_line)

        ax.errorbar(
            arr_cs_cc_numu_ggm_ps_energy_central,
            arr_cs_cc_numu_ggm_ps_cs_central,
            yerr=[arr_cs_cc_numu_ggm_ps_cs_lo, arr_cs_cc_numu_ggm_ps_cs_hi],
            linestyle="None",
            color="k",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="s",
        )  # , label='GGM-PS'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            markersize=int(0.8 * markersize),
            marker=r"s",
            label="GGM-PS 79",
        )
        col_lines.append(col_line)

    if "IHEP" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_ihep_itep_energy_central,
            arr_cs_cc_numu_ihep_itep_cs_central,
            yerr=[arr_cs_cc_numu_ihep_itep_cs_lo, arr_cs_cc_numu_ihep_itep_cs_hi],
            linestyle="None",
            color="k",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="v",
        )  # , label='IHEP-ITEP'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            markersize=markersize,
            marker=r"v",
            label="IHEP-ITEP 79",
        )
        col_lines.append(col_line)

        ax.errorbar(
            arr_cs_cc_numu_ihep_jinr_energy_central,
            arr_cs_cc_numu_ihep_jinr_cs_central,
            yerr=[arr_cs_cc_numu_ihep_jinr_cs_lo, arr_cs_cc_numu_ihep_jinr_cs_hi],
            linestyle="None",
            color="0.5",
            markeredgecolor="0.5",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="v",
        )  # , label='IHEP-JINR'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="0.5",
            markeredgecolor="0.5",
            markersize=markersize,
            marker=r"v",
            label="IHEP-JINR 96",
        )
        col_lines.append(col_line)

    if "NuTeV" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_nutev_energy_central,
            arr_cs_cc_numu_nutev_cs_central,
            xerr=[arr_cs_cc_numu_nutev_energy_lo, arr_cs_cc_numu_nutev_energy_hi],
            yerr=[arr_cs_cc_numu_nutev_cs_lo, arr_cs_cc_numu_nutev_cs_hi],
            linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="k",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="",
            marker="d",
            zorder=2,
        )  # , label='NuTeV'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="k",
            markersize=int(0.8 * markersize),
            marker="d",
            label="NuTeV 06",
        )
        col_lines.append(col_line)
        # print ('NuTeV nu', arr_cs_cc_numu_nutev_energy_central, arr_cs_cc_numu_nutev_cs_central, [arr_cs_cc_numu_nutev_cs_lo, arr_cs_cc_numu_nutev_cs_hi])

    if "FASERnu" in experiments:
        xerr = [arr_cs_cc_numu_fasernu_energy_lo, arr_cs_cc_numu_fasernu_energy_hi]
        fasernu_energy_central = arr_cs_cc_numu_fasernu_energy_central[0]
        fasernu_energy_min = fasernu_energy_central - xerr[0][0]
        fasernu_energy_max = fasernu_energy_central + xerr[1][0]
        fasernu_cross_section_central = arr_cs_cc_numu_fasernu_cs_central[0]

        if curved_fasernu_errorbar:
            # draw errorbar which follows expectation from bodek-yang, but is shifted so that
            # it goes through measured central value
            fasernu_errorbar_x = np.linspace(
                fasernu_energy_min, fasernu_energy_max, 100
            )
            fasernu_errorbar_y = get_cross_section_nu_nubar(
                lower=fasernu_energy_min,
                upper=fasernu_energy_max,
                num_steps=100,
                scaling=1.0,
            )
            xsec_sim = (
                get_cross_section_nu_nubar(lower=1000, num_steps=1, scaling=1.0)[0]
                / 1000
            )
            shift = fasernu_cross_section_central - xsec_sim
            ax.plot(
                fasernu_errorbar_x,
                fasernu_errorbar_y / fasernu_errorbar_x + shift,
                color="0.5",
                linewidth=elinewidth,
            )
            xerr = [[np.nan], [np.nan]]

        ax.errorbar(
            arr_cs_cc_numu_fasernu_energy_central,
            arr_cs_cc_numu_fasernu_cs_central,
            xerr=xerr,
            yerr=[arr_cs_cc_numu_fasernu_cs_lo, arr_cs_cc_numu_fasernu_cs_hi],
            linestyle="None",
            marker="x",
            color="0.5",
            mew=1,
            markerfacecolor="w",
            markersize=int(1.3 * markersize),
            elinewidth=elinewidth,
            zorder=2,
        )

        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            marker="x",
            color="0.5",
            mew=1,
            markerfacecolor="w",
            markersize=int(1.3 * markersize),
            label=r"FASER$\nu$ 24 $\nu_{\mu} + \bar{\nu}_{\mu}$",
        )
        lhc_lines.append(col_line)

    if "SciBoone" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_sciboone_energy_central,
            arr_cs_cc_numu_sciboone_cs_central,
            xerr=[arr_cs_cc_numu_sciboone_energy_lo, arr_cs_cc_numu_sciboone_energy_hi],
            yerr=[arr_cs_cc_numu_sciboone_cs_lo, arr_cs_cc_numu_sciboone_cs_hi],
            linestyle="None",
            color="k",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="x",
        )  # , label='SciBooNE'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            markersize=markersize,
            marker=r"x",
            label="SciBooNE 11",
        )
        col_lines.append(col_line)

    if "SKAT" in experiments:
        ax.errorbar(
            arr_cs_cc_numu_skat_energy_central,
            arr_cs_cc_numu_skat_cs_central,
            yerr=[arr_cs_cc_numu_skat_cs_lo, arr_cs_cc_numu_skat_cs_hi],
            linestyle="None",
            color="k",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="",
            marker=r"$\otimes$",
        )  # , label='SKAT'
        col_line = ax.plot(
            [1.0e-2, -1.0],
            linestyle="None",
            color="k",
            markersize=markersize,
            marker=r"$\otimes$",
            label="SKAT 79",
        )
        col_lines.append(col_line)

    # NUMUBAR

    if "Argoneut" in experiments:
        ax.errorbar(
            arr_cs_cc_numubar_argoneut_14_energy_central,
            arr_cs_cc_numubar_argoneut_14_cs_central,
            xerr=[
                arr_cs_cc_numubar_argoneut_14_energy_lo,
                arr_cs_cc_numubar_argoneut_14_energy_hi,
            ],
            yerr=[
                arr_cs_cc_numubar_argoneut_14_cs_lo,
                arr_cs_cc_numubar_argoneut_14_cs_hi,
            ],
            linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="^",
        )  # , label='ArgoNeuT 2014'

    if "CCFR" in experiments:
        ax.errorbar(
            arr_cs_cc_numubar_ccfr_energy_central,
            arr_cs_cc_numubar_ccfr_cs_central,
            yerr=[arr_cs_cc_numubar_ccfr_cs_lo, arr_cs_cc_numubar_ccfr_cs_hi],
            linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="",
            marker="d",
            zorder=2,
        )  # , label='CCFR'
        # print ('CCFR nubar', arr_cs_cc_numubar_ccfr_energy_central, arr_cs_cc_numubar_ccfr_cs_central, [arr_cs_cc_numubar_ccfr_cs_lo, arr_cs_cc_numubar_ccfr_cs_hi])

    if "CDHSW" in experiments:
        ax.errorbar(
            arr_cs_cc_numubar_cdhs_energy_central,
            arr_cs_cc_numubar_cdhs_cs_central,
            yerr=[arr_cs_cc_numubar_cdhs_cs_lo, arr_cs_cc_numubar_cdhs_cs_hi],
            # linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="w",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="--s",
        )  # , label='CDHS'
        # print ('CDHSW nubar', arr_cs_cc_numubar_cdhs_energy_central, arr_cs_cc_numubar_cdhs_cs_central, [arr_cs_cc_numubar_cdhs_cs_lo, arr_cs_cc_numubar_cdhs_cs_hi])

    if "GGM" in experiments:
        ax.errorbar(
            arr_cs_cc_numubar_ggm_sps_energy_central,
            arr_cs_cc_numubar_ggm_sps_cs_central,
            yerr=[arr_cs_cc_numubar_ggm_sps_cs_lo, arr_cs_cc_numubar_ggm_sps_cs_hi],
            linestyle="None",
            color="0.5",
            markeredgecolor="0.5",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="s",
        )  # , label='GGM-SPS'

        ax.errorbar(
            arr_cs_cc_numubar_ggm_ps_energy_central,
            arr_cs_cc_numubar_ggm_ps_cs_central,
            yerr=[arr_cs_cc_numubar_ggm_ps_cs_lo, arr_cs_cc_numubar_ggm_ps_cs_hi],
            linestyle="None",
            color="k",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="s",
        )  # label='GGM-PS',

    if "IHEP" in experiments:
        ax.errorbar(
            arr_cs_cc_numubar_ihep_itep_energy_central,
            arr_cs_cc_numubar_ihep_itep_cs_central,
            yerr=[arr_cs_cc_numu_ihep_itep_cs_lo, arr_cs_cc_numu_ihep_itep_cs_hi],
            linestyle="None",
            color="k",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="v",
        )  # label='IHEP-ITEP',

        ax.errorbar(
            arr_cs_cc_numubar_ihep_jinr_energy_central,
            arr_cs_cc_numubar_ihep_jinr_cs_central,
            yerr=[arr_cs_cc_numubar_ihep_jinr_cs_lo, arr_cs_cc_numubar_ihep_jinr_cs_hi],
            linestyle="None",
            color="0.5",
            markeredgecolor="0.5",
            markersize=markersize,
            elinewidth=elinewidth,
            fmt="v",
        )  # label='IHEP-JINR',

    if "MINOS" in experiments:
        ax.errorbar(
            arr_cs_cc_numubar_minos_energy_central,
            arr_cs_cc_numubar_minos_cs_central,
            yerr=[arr_cs_cc_numubar_minos_cs_lo, arr_cs_cc_numubar_minos_cs_hi],
            linestyle="None",
            color="k",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="o",
        )  # label='MINOS',

    if "NuTeV" in experiments:
        ax.errorbar(
            arr_cs_cc_numubar_nutev_energy_central,
            arr_cs_cc_numubar_nutev_cs_central,
            xerr=[arr_cs_cc_numubar_nutev_energy_lo, arr_cs_cc_numubar_nutev_energy_hi],
            yerr=[arr_cs_cc_numubar_nutev_cs_lo, arr_cs_cc_numubar_nutev_cs_hi],
            # linestyle="None",
            color="k",
            mew=1,
            markerfacecolor="k",
            markersize=int(0.8 * markersize),
            elinewidth=elinewidth,
            fmt="--",
            marker="d",
            zorder=0,
        )  # , label='NuTeV'
        # print ('NuTeV nubar', arr_cs_cc_numubar_nutev_energy_central, arr_cs_cc_numubar_nutev_cs_central, [arr_cs_cc_numubar_nutev_cs_lo, arr_cs_cc_numubar_nutev_cs_hi])

    return col_lines, lhc_lines, cosmic_lines


def plot_bodek_yank(
    plots: list[dict[str, Any]],
    ax: plt.Axes,
    divide_energy: bool = True,
    linewidth: float = 1.0,
    plot_uncertainty: bool = True,
    add_separation_sim: bool = True,
) -> list[mpl.lines.Line2D]:
    ret = []

    if add_separation_sim:
        # add vertical space between data points and simulation
        _label_line1 = ax.plot(np.nan, ls="", marker="", label=r"$\ $")
        ret.append(_label_line1)

    for p in plots:
        (
            _e,
            _xsec,
            _linestyle,
            _label,
            _color,
        ) = (
            p["E"],
            p["xsec"],
            p["ls"],
            p["label"],
            p["color"],
        )
        if divide_energy:
            _xsec /= _e
        # gainsboro is too light, use black instead
        if _color == "gainsboro":
            _linecolor = "black"
        else:
            _linecolor = _color
        (_line,) = ax.plot(
            _e,
            _xsec,
            ls=_linestyle,
            # c="black",
            c=_linecolor,
            label=_label,
            linewidth=linewidth,
        )
        if plot_uncertainty:
            _band = ax.fill_between(
                _e,
                _xsec * 0.94,
                _xsec * 1.06,
                # color="gainsboro",
                color=_color,
                alpha=0.4,
                linewidth=0,
                zorder=-1,
            )

        if _label != "":
            ret.append((_line, _band))

    # energy_centers = np.load(get_npy_path() / "nu_nubar_energies.npy")
    # cross_section_nu_nubar_by_energy = np.load(
    #     get_npy_path() / "cross_section_nu_nubar_by_energy_epos_p8.npy",
    # )
    # ax.plot(energy_centers, cross_section_nu_nubar_by_energy, color="green")

    return ret


def plot_cross_section(
    plots: list[dict[str, Any]],
    divide_energy: bool = True,
    show_bodek_yang: bool = True,
    bodek_yang_colors: list[str] | None = None,
    bodek_yang_linestyles: list[str] | None = None,
    bodek_yang_labels: list[str] | None = None,
    add_nu_labels: bool = True,
    add_separation_sim: bool = True,
    experiments_all: list[str] | None = None,
    experiments_dis: list[str] | None = None,
    title: str = "",
    title_left: str = r"${\bf FASER}$ Preliminary",
    title_right: str = rf"$\mathrm{{L}} = {get_luminosity():.1f}\;\mathrm{{fb}}^{{-1}}$",
    show_logo: bool = True,
    logo_text: str = "Preliminary",
    title_fontsize: int = 16,
    axis_fontsize: int = 12,
    label_fontsize: int = 14,
    markersize: int = 5,
    elinewidth: int = 1,
    capsize: int = 1,
    logo_scale: float = 1.0,
    logo_pos: tuple[float] = (0, 1.005),
    dir: str = ".",
    name: str | None = None,
    xlim: tuple[float] = (1e0, 3e5),
    fixed_target_upper_edge: float = 3.5e2,
    astro_lower_edge: float = 6e3,
    show_plot: bool = True,
    curved_fasernu_errorbar: bool = True,
) -> None:
    # setup plot style
    mpl.rcParams.update({"font.size": 13})
    mpl.rcParams["text.usetex"] = True
    fig, ax = plt.subplots(figsize=(6, 4))
    # setup some nice axes
    x0, x1 = 0.1, 0.96
    y0, y1 = 0.14, 0.82
    # xlo, xhi = 1e0, 3e5
    xlo, xhi = xlim
    if divide_energy:
        ylo, yhi = 1e-2, 2.5
    else:
        ylo, yhi = 3e-1, 2e5
    # ax = fig.add_axes((x0, y0, (x1 - x0), y1))
    # add gray shading in back
    ax.fill_between(
        [10**-1, fixed_target_upper_edge],
        [ylo, ylo],
        [yhi, yhi],
        color="gainsboro",
        zorder=-10,
    )
    ax.fill_between(
        [astro_lower_edge, 10**6], [ylo, ylo], [yhi, yhi], color="gainsboro", zorder=-10
    )

    # main existing constraints
    if experiments_all is None:
        experiments_all = [
            "T2K",
            "Argoneut",
            "ANL",
            "BEBC",
            "BNL",
            "CCFR",
            "CDHSW",
            "GGM",
            "IHEP",
            "MINOS",
            "NOMAD",
            "NuTeV",
            "SciBoone",
            "SKAT",
            "IceCube",
            "IceCube-HESE",
            "IceCube-HESE-20",
        ]
    if experiments_dis is None:
        experiments_dis = [
            "MINOS",
            "NOMAD",
            "NuTeV",
            "CCFR",
            "CDHSW",
            "IceCube",
            "IceCube-HESE",
            "IceCube-HESE-20",
        ]
    col_lines, lhc_lines, cosmic_lines = Plot_Cross_Section_Measurements(
        ax,
        experiments=experiments_all,
        divide_energy=divide_energy,
        markersize=markersize,
        elinewidth=elinewidth,
        print_xsref=False,
        curved_fasernu_errorbar=curved_fasernu_errorbar,
    )

    cur_lines = []
    for p in plots:
        if "markersize_scaling" in p:
            _markersize = markersize * np.array(p["markersize_scaling"])
        else:
            _markersize = markersize
        label, color, marker = p["label"], p["color"], p["marker"]
        (E, xsec, E_err_tmp, xsec_err_tmp) = (
            p["E"],
            p["xsec"],
            p["E_err"],
            np.array(p["xsec_err"]),
        )
        # E_err_tmp = np.reshape(E_err_tmp, (2,1))
        # xsec_err_tmp = np.reshape(xsec_err_tmp, (2,1))
        if divide_energy:
            _xsec = xsec
            _xsec_err = xsec_err_tmp
        else:
            _xsec = xsec * E
            _xsec_err = xsec_err_tmp * E
        cur_line = ax.errorbar(
            E,
            _xsec,
            xerr=E_err_tmp,
            yerr=_xsec_err,
            color=color,
            label=label,
            ls="",
            marker=marker,
            zorder=10,
            capsize=capsize,
            elinewidth=elinewidth,
            markersize=_markersize,
        )
        if p["label"] is not None:
            cur_lines.append(cur_line)

    if bodek_yang_labels is None:
        bodek_yang_labels = [
            r"Bodek-Yang, $\nu_{\mu}$",
            r"Bodek-Yang, $\bar{\nu}_{\mu}$",
            r"Bodek-Yang, $\nu_{\mu} + \bar{\nu}_{\mu}$",
        ]

    if show_bodek_yang is not None:
        bodek_yang_plots = get_bodek_yang_plots(
            upper=astro_lower_edge,
            colors=bodek_yang_colors,
            linestyles=bodek_yang_linestyles,
            labels=bodek_yang_labels,
        )
        by_plots = plot_bodek_yank(
            plots=bodek_yang_plots,
            ax=ax,
            divide_energy=divide_energy,
            add_separation_sim=add_separation_sim,
        )
    if add_nu_labels:
        # label_positions = [(1.2e3, 1.1), (7e2, 0.18), (2e3, 0.38)]
        label_positions = [(0.9e3, 1.25), (7e2, 0.185), (2e3, 0.38)]
        label_texts = [
            r"$\nu_{\mu}$",
            r"$\bar{\nu}_{\mu}$",
            r"$\nu_{\mu} + \bar{\nu}_{\mu}$",
        ]
        label_colors = [plot["color"] for plot in plots[:3]]
        # label_fontsizes = [title_fontsize, title_fontsize, 0.8 * title_fontsize]
        label_fontsizes = [axis_fontsize] * 3
        for _pos, _label_text, _color, _fontsize in zip(
            label_positions, label_texts, label_colors, label_fontsizes
        ):
            ax.text(_pos[0], _pos[1], _label_text, color=_color, fontsize=_fontsize)
        # ax.text(
        #     1.2e3,
        #     1.1,
        #     r"$\nu_{\mu}$",
        #     color=TangoColors.scarlet_red,
        #     fontsize=title_fontsize,
        # )
        # ax.text(
        #     7e2,
        #     0.18,
        #     r"$\bar{\nu}_{\mu}$",
        #     color=TangoColors.orange,
        #     fontsize=title_fontsize,
        # )
        # ax.text(
        #     2e3,
        #     0.38,
        #     r"$\nu_{\mu} + \bar{\nu}_{\mu}$",
        #     color=TangoColors.sky_blue,
        #     fontsize=0.8 * title_fontsize,
        # )
    # finalize layout
    ax.set_xlim(xlo, xhi)
    # ax.set_ylim([ylo, yhi])
    ax.set_xlabel("Neutrino energy [GeV]", fontsize=axis_fontsize)
    if divide_energy:
        ax.set_ylabel(
            r"$\sigma_{\mathsf{CC}} \, / \, E_{\nu} \ \mathsf{[ \, 10^{-38} \ cm^2 \, / \, GeV \, / \, nucleon \, ]}$",
            fontsize=axis_fontsize,
        )
    else:
        ax.set_ylabel(
            r"$\sigma_{\mathsf{CC}} \ \mathsf{[\, 10^{-38} \ cm^2 \, / \, nucleon \,]}$",
            fontsize=axis_fontsize,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # labels
    # if divide_energy:
    #     _nu_label_pos = (0.30, 0.79)
    #     _nu_bar_label_pos = (0.30, 0.47)
    # else:
    #     _nu_label_pos = (0.35, 0.43)
    #     _nu_bar_label_pos = (0.35, 0.26)
    # ax.annotate(
    #     r"$\nu$",
    #     xy=_nu_label_pos,
    #     xycoords="axes fraction",
    #     color="k",
    #     fontsize=title_fontsize,
    #     ha="left",
    # )
    # ax.annotate(
    #     r"$\bar{\nu}$",
    #     xy=_nu_bar_label_pos,
    #     xycoords="axes fraction",
    #     color="k",
    #     fontsize=title_fontsize,
    #     ha="left",
    # )
    # ax.set_ylim(1e-2, 4)
    ax.set_ylim(ylo, yhi)
    # accelerator, LHC, astro label
    for _label, _label_pos in zip(
        [r"Accelerator $\nu_{\mu}$", r"Collider $\nu_{\mu}$", r"Astro $\nu_{\mu}$"],
        [
            get_log_mean(xlo, fixed_target_upper_edge),
            get_log_mean(fixed_target_upper_edge, astro_lower_edge),
            get_log_mean(astro_lower_edge, xhi),
        ],
    ):
        ax.annotate(
            _label,
            xy=(_label_pos, 0.92 * yhi),
            color="k",
            fontsize=axis_fontsize,
            horizontalalignment="center",
            verticalalignment="top",
        )
    # accelerator legends
    lines = [line[0] for line in col_lines]
    labels = [line[0].get_label() for line in col_lines]
    if divide_energy:
        _legend_pos = "lower left"
        _bbox_to_anchor = [-0.02, 0.0]
    else:
        _legend_pos = "upper left"
        _bbox_to_anchor = [0.0, 0.90]
    legend_1 = ax.legend(
        lines,
        labels,
        loc=_legend_pos,
        numpoints=1,
        ncol=2,
        columnspacing=0.5,
        frameon=False,
        fontsize=8.5,
        borderpad=0.9,
        handlelength=0,
        bbox_to_anchor=_bbox_to_anchor,
    )

    # LHC legends
    # lines = [line[0] for line in lhc_lines]
    # labels = [line[0].get_label() for line in lhc_lines]
    # if divide_energy:
    #     _legend_pos = "lower center"
    #     _bbox_to_anchor = [0.0, 0.5]
    # legend_4 = ax.legend(
    #     lines,
    #     labels,
    #     loc=_legend_pos,
    #     numpoints=1,
    #     ncol=2,
    #     columnspacing=0.5,
    #     frameon=False,
    #     fontsize=label_fontsize,
    #     borderpad=0.9,
    #     handlelength=0,
    #     bbox_to_anchor=_bbox_to_anchor,
    # )

    # cosmic legends
    lines = [line[0] for line in cosmic_lines]
    labels = [line[0].get_label() for line in cosmic_lines]
    legend_2 = ax.legend(
        lines,
        labels,
        loc="lower right",
        numpoints=1,
        columnspacing=0.5,
        frameon=False,
        fontsize=8.5,
        borderpad=0.9,
        handlelength=2,
        bbox_to_anchor=[1.03, 0.0],
    )

    # FASER
    lines = [line for line in cur_lines]
    labels = [line.get_label() for line in cur_lines]

    # FASERnu
    lines += [line[0] for line in lhc_lines]
    labels += [line[0].get_label() for line in lhc_lines]

    # Bodek-Yang
    if bodek_yang_plots is not None:
        lines += [
            line if line[0].get_label().startswith("Bodek-Yang") else line[0]
            for line in by_plots
        ]
        labels += [line[0].get_label() for line in by_plots]

    if not any(bodek_yang_labels):
        # Add empty line in legend to separate simulation from data
        # (_label_line1,) = ax.plot(np.nan, ls="", marker="")

        # Add default Bodek-Yang label
        (_label_line2,) = ax.plot([np.nan], [np.nan], color="black", ls="--")
        _label_line3 = ax.fill_between(
            x=[np.nan],
            y1=[np.nan],
            y2=[np.nan],
            color="gainsboro",
            alpha=0.5,
            linewidth=0,
            zorder=-1,
        )
        # lines += [_label_line1, (_label_line2, _label_line3)]
        # labels += [r"$\ $", "Bodek-Yang sim."]
        lines += [(_label_line2, _label_line3)]
        labels += ["Bodek-Yang sim."]

    legend_3 = ax.legend(
        lines,
        labels,
        loc="lower center",
        numpoints=1,
        columnspacing=0.5,
        frameon=False,
        fontsize=8.5,
        borderpad=0.9,
        handlelength=1.7,
        handleheight=1,
        bbox_to_anchor=[0.545, 0.0],
    )
    ax.add_artist(legend_1)
    ax.add_artist(legend_2)
    ax.add_artist(legend_3)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_title(title_right, loc="right", fontsize=title_fontsize)
    if divide_energy:
        name = f"{name}_norm"
    if show_logo:
        plt.tight_layout()
        add_logo(
            ax=ax,
            text=logo_text,
            text_fontsize=12,
            scale=logo_scale,
            xy=logo_pos,
        )
        if name is not None:
            save(
                dir=get_figure_path() / dir,
                name=name,
                show_plot=show_plot,
                show_logo=True,
            )
        else:
            if show_plot:
                show()
            else:
                plt.close()
    else:
        with plt.rc_context({"text.usetex": True}):
            ax.set_title(title_left, loc="left")
        if name is not None:
            plt.savefig(f"{name}.pdf")
        if show_plot:
            plt.show()
        else:
            plt.close()

    # if name is not None:
    #     plt.savefig(get_figure_path() / dir / f"{name}.pdf")
    # plt.show()
