function [TairNext, TmassNext] = rc_thermal_step(Tair, Tmass, Tout, Phvac, Qinternal, Qsolar, p, dt)
%RC_THERMAL_STEP One discrete step of the two-state RC thermal model.
% p fields: r_ao, r_am, c_air, c_mass, hvac_efficiency
% Positive Phvac adds heat; negative Phvac represents cooling.

if nargin < 8
    dt = 1.0;
end
airFlow = (Tout - Tair) / p.r_ao;
massFlow = (Tmass - Tair) / p.r_am;
hvacGain = p.hvac_efficiency * Phvac;
TairNext = Tair + dt * (airFlow + massFlow + hvacGain + Qinternal + Qsolar) / p.c_air;
TmassNext = Tmass + dt * (Tair - Tmass) / (p.r_am * p.c_mass);
end
