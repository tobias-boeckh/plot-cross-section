from abc import ABC, abstractmethod

import numpy as np
import uproot


def _interpolate_cross_section(
    energy: float | np.ndarray,
    pdg: float | np.ndarray,
    energy_nu: np.ndarray,
    sigma_nu: np.ndarray,
    energy_nu_bar: np.ndarray,
    sigma_nu_bar: np.ndarray,
):
    good_pdg_ids = [12, -12, 14, -14, 16, -16]
    if np.any(~np.isin(pdg, good_pdg_ids)):
        raise ValueError(
            f"PDG ids have to be in {', '.join([str(i) for i in good_pdg_ids])}."
        )
    return np.where(
        pdg > 0,
        np.interp(energy, energy_nu, sigma_nu, left=np.nan, right=np.nan),
        np.interp(energy, energy_nu_bar, sigma_nu_bar, left=np.nan, right=np.nan),
    )


class CrossSectionWrapper(ABC):
    @abstractmethod
    def get_cross_section(
        self,
        energy: float | np.ndarray,
        pdg: float | np.ndarray,
    ) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def energy_nu(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def energy_nu_bar(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def sigma_nu(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def sigma_nu_bar(self) -> np.ndarray:
        pass


class FaserCrossSectionWrapper(CrossSectionWrapper):
    def __init__(self, path: str | None = None) -> None:
        if path is None:
            self.__path = "data/faser-neutrino-tungsten-cross-section.root"
        else:
            self.__path = path

        self._energy_nu, self._sigma_nu = uproot.open(self.__path)[
            "nu_mu_W184/tot_cc"
        ].values(axis="both")  # energy in GeV, sigma in 10^{-38} cm^2
        self._energy_nu_bar, self._sigma_nu_bar = uproot.open(self.__path)[
            "nu_mu_bar_W184/tot_cc"
        ].values(axis="both")
        # get the cross-section per nucleon
        a = 184
        self._sigma_nu /= a
        self._sigma_nu_bar /= a

    def get_cross_section(
        self, energy: float | np.ndarray, pdg: float | np.ndarray
    ) -> np.ndarray:
        """Get Bodek-Yang cross section for given energy and pdg id."""
        return _interpolate_cross_section(
            energy=energy,
            pdg=pdg,
            energy_nu=self._energy_nu,
            sigma_nu=self.sigma_nu,
            energy_nu_bar=self._energy_nu_bar,
            sigma_nu_bar=self._sigma_nu_bar,
        )

    @property
    def energy_nu(self) -> np.ndarray:
        return self._energy_nu

    @property
    def sigma_nu(self) -> np.ndarray:
        return self._sigma_nu

    @property
    def energy_nu_bar(self) -> np.ndarray:
        return self._energy_nu_bar

    @property
    def sigma_nu_bar(self) -> np.ndarray:
        return self._sigma_nu_bar
