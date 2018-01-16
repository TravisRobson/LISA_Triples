#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Constants.h"
#include "Triple.h"
#include "LISA.h"
#include "GB.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>


double get_u(double l, double e)
{
	///////////////////////
	//
	// Invert Kepler's equation l = u - e sin(u)
	// Using Mikkola's method (1987)
	// referenced Tessmer & Gopakumar 2007
	//
	///////////////////////
	
	double u0;							// initial guess at eccentric anomaly
	double z, alpha, beta, s, w;		// auxiliary variables
	double mult;						// multiple number of 2pi
	
	int neg		 = 0;					// check if l is negative
	int over2pi  = 0;					// check if over 2pi
	int overpi	 = 0;					// check if over pi but not 2pi
	
	double f, f1, f2, f3, f4;			// pieces of root finder
	double u, u1, u2, u3, u4;
	
	// enforce the mean anomaly to be in the domain -pi < l < pi
	if (l < 0)
	{
		neg = 1;
		l   = -l;
	}
	if (l > 2.*M_PI)
	{
		over2pi = 1;
		mult	= floor(l/(2.*M_PI));
		l	   -= mult*2.*M_PI;
	}
	if (l > M_PI)
	{
		overpi = 1;
		l	   = 2.*M_PI - l;
	}
	
	alpha = (1. - e)/(4.*e + 0.5);
	beta  = 0.5*l/(4.*e + 0.5);
	
	z = sqrt(beta*beta + alpha*alpha*alpha);
	if (neg == 1) z = beta - z;
	else	      z = beta + z;
	
	// to handle nan's from negative arguments
	if (z < 0.) z = -pow(-z, 0.3333333333333333);
	else 	    z =  pow( z, 0.3333333333333333);
	
	s  = z - alpha/z;
	w  = s - 0.078*s*s*s*s*s/(1. + e);
	
	u0 = l + e*(3.*w - 4.*w*w*w);
	
	// now this initial guess must be iterated once with a 4th order Newton root finder
	f  = u0 - e*sin(u0) - l;
	f1 = 1. - e*cos(u0);
	f2 = u0 - f - l;
	f3 = 1. - f1;
	f4 = -f2;
	
	f2 *= 0.5;
	f3 *= 0.166666666666667;
	f4 *= 0.0416666666666667;
	
	u1 = -f/f1;
	u2 = -f/(f1 + f2*u1);
	u3 = -f/(f1 + f2*u2 + f3*u2*u2);
	u4 = -f/(f1 + f2*u3 + f3*u3*u3 + f4*u3*u3*u3);
	
	u = u0 + u4;
	
	if (overpi  == 1) u = 2.*M_PI - u;
	if (over2pi == 1) u = 2.*M_PI*mult + u;
	if (neg		== 1) u = -u;

	return u;
}

double get_phi(double t, double T, double e, double n)
{
	double u, beta;
	
	u = get_u(n*(t-T), e);
	
	if (e == 0.) return u;
	
	beta = (1. - sqrt(1. - e*e))/e;
	
	return u + 2.*atan2( beta*sin(u), 1. - beta*cos(u));
}

double get_vLOS(struct Triple *trip, double t)
{
 	double phi2;
	double T2, e2, n2;
	double omegabar;
	double A2;
	
	A2       = exp(trip->params[0]);
	omegabar = trip->params[1];
	e2       = trip->params[2];
	n2       = exp(trip->params[3])/YEAR;
	T2       = YEAR*exp(trip->params[4]);
			
	phi2 = get_phi(t, T2, e2, n2);
	
	return A2*(sin(phi2 + omegabar) + e2*sin(omegabar));
}

void solve_phase_H(struct Triple *trip, struct GB *gb, double **soln, long N, double T)
{
	long i;		    // iterator, number of samples
	
	double dtt;
	double sum, t;

	dtt = T/(double)(N-1);	
	
	// I will perform a parabolic integration
	sum = parab_step(trip, gb, 0., dtt);
	soln[0][0] = 0.;	// initial time
	soln[1][0] = 0.;    // initial phase
	
	for (i=1; i<N; i++)
	{
		t = (double)i*dtt;
		sum += parab_step(trip, gb, t, dtt);
		soln[0][i] = t;
		soln[1][i] = sum;
	}
	
	return;
}	

double get_fGW(struct GB *gb, double t)
{
	double f0, dfdt_0, d2fdt2_0;
	
	f0        = gb->params[0]/gb->T;
	dfdt_0    = gb->params[7]/gb->T/gb->T;
	d2fdt2_0  = 11./3.*dfdt_0*dfdt_0/f0;
	
	// assuming t0 = 0.
	return f0 + dfdt_0*t + 0.5*d2fdt2_0*t*t;
}

double parab_step(struct Triple *trip, struct GB *gb, double t0, double dtt)
{
	// step in an integral using parabolic approximation to integrand
	// g1 starting point
	// g2 mid-point
	// g3 end-point
	
	double g1, g2, g3;
	
	g1 = get_vLOS(trip, t0)*get_fGW(gb, t0);
	g2 = get_vLOS(trip, t0 + 0.5*dtt)*get_fGW(gb, t0 + 0.5*dtt);
	g3 = get_vLOS(trip, t0 + dtt)*get_fGW(gb, t0 + dtt);
	
	return 0.166666666666666666667*dtt*(g1 + g3 + 4.0*g2)*PI2/C;
}

void GB_triple(struct GB *gb, double *XLS, double *ALS, double *ELS, struct Triple *trip)
{
	int n;      // iterator
	long No;	// number of samples for triple phase			
	double t;	// time
	double new_f0;
	
	// waveform struct to hold pieces for calculation
	struct Waveform *wfm = malloc(sizeof(struct Waveform));
	
	wfm->N = gb->N;				     // set number of samples
	wfm->T = gb->T;  		         // set observation period
	wfm->NP = 8;					 // just the relevant GB ones 
	alloc_waveform(wfm);		     // allocate memory to hold pieces of waveform
	copy_params(wfm, gb->params);    // copy parameters to waveform structure
	
	get_basis_tensors(wfm);          //  Tensor construction for building slowly evolving LISA response  
	
	// HACK AWAYYYY
	No = 10000;
	
	double **phase_H_soln;
	
	phase_H_soln    = malloc(2*sizeof(double *));
	phase_H_soln[0] = malloc(No*sizeof(double));
	phase_H_soln[1] = malloc(No*sizeof(double)); 
	
	solve_phase_H(trip, gb, phase_H_soln, No, wfm->T);
	trip->phase = phase_H_soln;
	
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
 	gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, No);
 	trip->spline = spline;
 	trip->acc    = acc;
 	gsl_spline_init(trip->spline, phase_H_soln[0], phase_H_soln[1], No);

	// adjust the carrier bin
	new_f0 = gb->params[0]*(1. + get_vLOS(trip, 0.)/C);
	wfm->q = gb->q;

	for(n=0; n<wfm->N; n++)
	{
		t = wfm->T*(double)(n)/(double)wfm->N; // First time sample must be at t=0 for phasing

		calc_xi_f_trip(wfm, t, trip);     // calc frequency and time variables
		calc_sep_vecs(wfm);          // calculate the S/C separation vectors
		calc_d_matrices(wfm);        // calculate pieces of waveform
		calc_kdotr(wfm);		     // calculate dot product
		get_transfer_trip(wfm, t, trip);  // Calculating Transfer function

		fill_time_series(wfm, n);    // Fill  time series data arrays with slowly evolving signal. 
	}

	fft_data(wfm);     // Numerical Fourier transform of slowly evolving signal   
	unpack_data(wfm);  // Unpack arrays from FFT and normalize
	
	XYZ(wfm->d, new_f0/wfm->T, wfm->q, wfm->N, XLS, ALS, ELS); // Calculate other TDI channels
 
	free_waveform(wfm);  // Deallocate memory
	free(wfm);
	
	gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    
	free(phase_H_soln[0]);
	free(phase_H_soln[1]);
	return;
}

void calc_xi_f_trip(struct Waveform *wfm, double t, struct Triple *trip)
{
	long i;
	
	double f0, dfdt_0, d2fdt2_0;
	
	f0       = wfm->params[0]/wfm->T;
	dfdt_0   = wfm->params[7]/wfm->T/wfm->T;
	d2fdt2_0 = 11./3.*dfdt_0*dfdt_0/f0;
	
	spacecraft(t, wfm->x, wfm->y, wfm->z); // Calculate position of each spacecraft at time t

	for(i=0; i<3; i++) 
	{	
		wfm->kdotx[i] = (wfm->x[i]*wfm->k[0] + wfm->y[i]*wfm->k[1] + wfm->z[i]*wfm->k[2])/C;
		
		//Wave arrival time at spacecraft i
		wfm->xi[i]    = t - wfm->kdotx[i];
		
		//First order approximation to frequency at spacecraft i
		wfm->f[i]     = f0 + dfdt_0*wfm->xi[i] + 0.5*d2fdt2_0*wfm->xi[i]*wfm->xi[i];
		wfm->f[i]    *= (1. + get_vLOS(trip, wfm->xi[i])/C);

		//Ratio of true frequency to transfer frequency
		wfm->fonfs[i] = wfm->f[i]/fstar;
	}
	
	return;
}

void get_transfer_trip(struct Waveform *wfm, double t, struct Triple *trip)
{
	long i, j;
	long q;
	
	double tran1r, tran1i;
	double tran2r, tran2i; 
	double aevol;			// amplitude evolution factor
	double arg1, arg2, sinc;
	double f0, dfdt_0, d2fdt2_0, phi0;
	double df;
	
	f0       = wfm->params[0]/wfm->T;
	phi0     = wfm->params[6];
	dfdt_0   = wfm->params[7]/wfm->T/wfm->T;
	d2fdt2_0 = 11./3.*dfdt_0*dfdt_0/f0;
	
	q  = wfm->q;
	df = PI2*(((double)q)/wfm->T);
		
	for(i=0; i<3; i++)
	{
		for(j=0; j<3; j++)
		{
			if(i!=j)
			{
				//fprintf(stdout, "xi[i]: %e\n", wfm->xi[i]);
				if (wfm->xi[i] < 0) wfm->xi[i] = 0.;	// Hack out the ass!
				
				//Argument of transfer function
				arg1 = 0.5*wfm->fonfs[i]*(1 - wfm->kdotr[i][j]);
				
				//Argument of complex exponentials
				arg2  = PI2*f0*wfm->xi[i] + M_PI*dfdt_0*wfm->xi[i]*wfm->xi[i] 
				       +M_PI*d2fdt2_0*wfm->xi[i]*wfm->xi[i]*wfm->xi[i]/3.0 
				       +phi0 - df*t;
				
				arg2 += gsl_spline_eval(trip->spline, wfm->xi[i], trip->acc); 
				
				//fprintf(stdout, "%e %e\n", gsl_spline_eval(trip->spline, wfm->xi[i], trip->acc), arg2);

			
				sinc = 0.25*sin(arg1)/arg1; //Transfer function
				
				//Evolution of amplitude
				aevol = 1.0 + 0.66666666666666666666*dfdt_0/f0*wfm->xi[i];  
				
				///Real and imaginary pieces of time series (no complex exponential)
				tran1r = aevol*(wfm->dplus[i][j]*wfm->DPr + wfm->dcross[i][j]*wfm->DCr);
				tran1i = aevol*(wfm->dplus[i][j]*wfm->DPi + wfm->dcross[i][j]*wfm->DCi);
				
				//Real and imaginry components of complex exponential
				tran2r = cos(arg1 + arg2);
				tran2i = sin(arg1 + arg2);    
				
				//Real & Imaginary part of the slowly evolving signal
				wfm->TR[i][j] = sinc*(tran1r*tran2r - tran1i*tran2i);
				wfm->TI[i][j] = sinc*(tran1r*tran2i + tran1i*tran2r);
			}
		}
	}
	
	return;
}