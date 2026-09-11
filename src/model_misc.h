/*
 * model_misc.h -- galaxy utilities, plus umbrella include for the physics
 * helper modules that were split out of model_misc.c.
 *
 * Declares galaxy initialisation, disk/bulge size calculations, and small
 * shared helpers, and pulls in model_regimes.h (CGM/FFB classification),
 * model_halo_properties.h (virial properties, concentration), and
 * model_h2_chemistry.h (H2 fraction prescriptions) so existing consumers
 * keep a single include.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    #include "model_h2_chemistry.h"
    #include "model_halo_properties.h"
    #include "model_regimes.h"

    /*
     * Divisors for the per-substep Sfr* accumulators.
     *
     * The Sfr* arrays hold STEPS fixed bins but receive one entry per substep, and
     * evolve_galaxies() integrates over SubstepsUsed substeps, which the adaptive scheme
     * and SubstepResolution can push above or below STEPS. Several substeps then land in
     * the same bin, so the bin sum is a sum over *substeps*, and the mean rate across the
     * snapshot is that sum divided by SubstepsUsed. Dividing by STEPS instead reports an
     * SFR scaled by SubstepsUsed / STEPS -- verified on Millennium as exactly 0.5x, 2.0x
     * and 3.0x for SubstepResolution 0.5, 2.0 and 3.0.
     *
     * The ColdGas/metals accumulators are assignments rather than sums, so exactly
     * min(SubstepsUsed, STEPS) distinct bins ever carry a value; that count is their
     * divisor. Both reduce to STEPS when SubstepsUsed == STEPS, which is every run where
     * the adaptive path does not fire, so default output is unchanged.
     */
    static inline int sfr_rate_divisor(const int substeps_used)
    {
        return (substeps_used > 0) ? substeps_used : STEPS;
    }

    static inline int sfr_metallicity_divisor(const int substeps_used)
    {
        const int n = sfr_rate_divisor(substeps_used);
        return (n < STEPS) ? n : STEPS;
    }

    /*
     * Deposit metals (or gas + metals) into a galaxy's hot-phase reservoir,
     * routed by regime: CGMgas when the CGM recipe is active and the galaxy
     * is CGM-regime (Regime == 0), HotGas otherwise. These helpers replace
     * the identical if/else ladder previously repeated at every deposit site;
     * the += operations are unchanged, so results are bit-identical.
     */
    static inline void add_metals_to_hot_reservoir(struct GALAXY *central, const struct params *run_params, const double metals)
    {
        if(run_params->CGMrecipeOn == 1 && central->Regime == 0) {
            central->MetalsCGMgas += metals;
        } else {
            central->MetalsHotGas += metals;
        }
    }

    static inline void add_gas_to_hot_reservoir(struct GALAXY *central, const struct params *run_params, const double gas, const double metals)
    {
        if(run_params->CGMrecipeOn == 1 && central->Regime == 0) {
            central->CGMgas += gas;
            central->MetalsCGMgas += metals;
        } else {
            central->HotGas += gas;
            central->MetalsHotGas += metals;
        }
    }

    /*
     * SF prescriptions that track molecular gas (H2gas/H1gas): every
     * prescription except 0 (Croton+06 cold-gas threshold) and 2
     * (Somerville+25 cold-gas efficiency). SFprescription is validated to
     * [0, 7] at parameter-read time, so the two-term form is exact.
     */
    static inline int sf_prescription_tracks_h2(const int sfprescription)
    {
        return sfprescription != 0 && sfprescription != 2;
    }

    /* BR06 pressure-based H2 family: 1 (BR06) and 3 (Somerville+25 + BR06). */
    static inline int sf_prescription_is_br06(const int sfprescription)
    {
        return sfprescription == 1 || sfprescription == 3;
    }

    /*
     * Effective supernova energy coupling for the FIRE ejection term.
     *
     * The FIRE branch sets E_FB = eps_halo * f_FIRE * 0.5 * m_* * eta_SN E_SN, so
     * the coupling eps_eff = FeedbackEjectionEfficiency * f_FIRE inherits the FIRE
     * redshift and velocity scaling and is unbounded: at low V_vir and high z it
     * exceeds 2, i.e. the ejection term spends more than the total supernova energy
     * m_* eta_SN E_SN released by the stars that drive it.
     *
     * With SNEnergyConservationOn = 1 the coupling is capped at MaxSNEnergyCoupling
     * (default 2.0, i.e. E_FB <= the whole SN budget; 1.0 caps it at half).  This is
     * an energy-conservation bound, not a tuning knob -- it constrains the model to
     * the energy actually available rather than truncating the empirical FIRE
     * mass-loading scaling, which is applied unmodified in eta_reheat.
     *
     * The bound is on by default.  Setting SNEnergyConservationOn = 0 restores the
     * unbounded coupling of the earlier published SAGE26 behaviour.
     */
    static inline double sn_energy_coupling(const double fire_scaling, const struct params *run_params)
    {
        const double eps_eff = run_params->FeedbackEjectionEfficiency * fire_scaling;
        if(run_params->SNEnergyConservationOn && eps_eff > run_params->MaxSNEnergyCoupling) {
            return run_params->MaxSNEnergyCoupling;
        }
        return eps_eff;
    }

    /*
     * Optional energy-conservation bound on the reheating term.
     *
     * mdot_reheat = eta_reheat * mdot_* carries no energy check of its own: the
     * FIRE scaling is applied unmodified however much energy the resulting
     * outflow would require.  Where the model ejects, the total energy spent is
     * exactly E_FB and is already bounded by sn_energy_coupling(); the residual
     * is the non-ejecting regime (V_vir above the ejection threshold), where the
     * reheating cost 0.5*eta_reheat*V_vir^2 -- the same convention as E_lift --
     * grows linearly with V_vir and is unbounded.
     *
     * With SNEnergyConservationOn = 1 the mass loading is capped so that this
     * cost cannot exceed the same MaxSNEnergyCoupling budget used for the
     * ejection term.  The bound is on by default; setting
     * SNEnergyConservationOn = 0 restores the unbounded mass loading.
     */
    static inline double capped_eta_reheat(const double eta_reheat, const double vvir,
                                           const struct params *run_params)
    {
        if(!run_params->SNEnergyConservationOn || vvir <= 0.0) {
            return eta_reheat;
        }
        const double esn_per_mass = run_params->EtaSNcode * run_params->EnergySNcode;
        const double eta_max = run_params->MaxSNEnergyCoupling * esn_per_mass / (vvir * vvir);
        return (eta_reheat > eta_max) ? eta_max : eta_reheat;
    }

    /* functions in model_misc.c */
    extern void init_galaxy(const int p, const int halonr, int *galaxycounter, const struct halo_data *halos, struct GALAXY *galaxies, const struct params *run_params);
    extern double get_metallicity(const double gas, const double metals);
    extern double get_disk_radius(const int halonr, const int p, const struct halo_data *halos, const struct GALAXY *galaxies);
    extern double get_bulge_radius(const int p, struct GALAXY *galaxies, const struct params *run_params);
    extern double dmax(const double x, const double y);
    extern void update_instability_bulge_radius(const int p, const double delta_mass,
                                     const double old_disk_radius,
                                     struct GALAXY *galaxies, const struct params *run_params);


#ifdef __cplusplus
}
#endif
