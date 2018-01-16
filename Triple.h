#ifndef triple_h
#define triple_h

#include <gsl/gsl_spline.h>
#include "GB.h"

struct Triple
{
	long NP;						// number of parameters
	
	// intrinsic
	double a2, e2, n2;				// semi-major axis, eccentricity, mean motion
	double iota2, Omega2, omega2;   // inclination, longitude of ascending node, longitude of pericenter
	double T2; 						// time of periapsis passage
	double m2, mc;				    // mass of all 3 bodies, mass of companion
	double phibar, omegabar;		// phase parameter
	double A2;						// amplitude factor
	
 	// derived
	double f, g;					// f = cos(theta)sin(iota2) + sin(theta)cos(iota2)cis(phi-Omega2)
									// g = sin(theta)cos(phi-Omega2)
														
	double *params;					// hold parameters (in dimensionless and potentially log form)
	
	double **phase;					// phase solution
	
	gsl_spline *spline;
	gsl_interp_accel *acc;
};

double get_u(double l, double e);
double get_phi(double t, double T, double e, double n);
double get_vLOS(struct Triple *trip, double t);

void GB_triple(struct GB *gb, double *XLS, double *ALS, double *ELS, struct Triple *trip);
void solve_phase_H(struct Triple *trip, struct GB *gb, double **soln, long N, double T);
double parab_step(struct Triple *trip, struct GB *gb, double t0, double dt);
void calc_xi_f_trip(struct Waveform *wfm, double t, struct Triple *trip);
void get_transfer_trip(struct Waveform *wfm, double t, struct Triple *trip);
double get_fGW(struct GB *gb, double t);


#endif /* triple_h */