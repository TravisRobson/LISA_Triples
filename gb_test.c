#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>

#include "Constants.h"
#include "LISA.h"
#include "GB.h"

long get_N(double *params);

int main(int argc, char *argv[])
{
	fprintf(stdout, "==============================================================\n\n");
	long i;        // iterator
	long N;        // number of samples to take of GB signal
	long NFFT;     // number of samples associated with entire signal (based on Tobs and sampling rate dt)
	long q;	       // carrier frequency bin of the GB
	long k;	       // frequency bin number
	long iRe, iIm;
	
	int NP;	       // number of parameters

	double f0, dfdt_0;
	double co_lat, phi;
	double psi, phi0, iota, amp;
	double f;
	
	double *params;
	double *XX, *AA, *EE;
	double *XXLS, *AALS, *EELS;
	double sqTobs;
	
	FILE *in_file, *out_file;
	
	NP = 9;
	sqTobs = sqrt(Tobs);
	
	params = malloc(NP*sizeof(double));
	
	in_file = fopen("test_src.dat", "r");
	
	fscanf(in_file, "%lg",   &f0);
	fscanf(in_file, "%lg",   &dfdt_0);
	fscanf(in_file, "%lg",   &co_lat);
	fscanf(in_file, "%lg",   &phi);
	fscanf(in_file, "%lg",   &amp);
	fscanf(in_file, "%lg",   &iota);
	fscanf(in_file, "%lg",   &psi);
	fscanf(in_file, "%lg",   &phi0);
	
	fclose(in_file);

	params[0] = f0*Tobs;
	params[1] = cos(PIon2 - co_lat); // convert to spherical polar
	params[2] = phi;
	params[3] = log(amp);
	params[4] = cos(iota);
	params[5] = psi;
	params[6] = phi0;
	params[7] = dfdt_0*Tobs*Tobs;
	params[8] = 11./3.*dfdt_0*dfdt_0/f0*Tobs*Tobs*Tobs; // assume GR radiation reaction
	
	q = (long)(params[0]); // carrier frequency bin
	
	fprintf(stdout,"q: %ld\n",q);

	N  = 1*get_N(params);
	fprintf(stdout, "No samples: %ld\n", N);
	
	XX = malloc(2*N*sizeof(double));
	AA = malloc(2*N*sizeof(double));
	EE = malloc(2*N*sizeof(double));

	FAST_LISA(params, N, XX, AA, EE, NP);
		
	out_file = fopen("test_signal.dat" ,"w");
	for (i=0; i<N; i++)
	{
		k = (q + i - N/2);    // frequency bin associated with ith sample in GB data arrays
		f = (double)(k)/Tobs; // convert bin number to frequency
		
		iRe = 2*i;
		iIm = 2*i+1;
		
		fprintf(out_file, "%.12g %e %e %e %e %e %e\n", f,  sqTobs*XX[iRe], sqTobs*XX[iIm],
														   sqTobs*AA[iRe], sqTobs*AA[iIm],
													       sqTobs*EE[iRe], sqTobs*EE[iIm]);
	}
	fclose(out_file);
	
	NFFT = (long)(Tobs/dt);
	fprintf(stdout, "NFFT: %ld, 2^{p}, p = %ld\n", NFFT, (long)(log(NFFT)/log(2.)));
	
	// arrays to store entire LISA signal, to demonstrate how FAST_LISA output
	//		is to dropped into the data stream
	XXLS = malloc(NFFT*sizeof(double));
	AALS = malloc(NFFT*sizeof(double));
	EELS = malloc(NFFT*sizeof(double));
	
	for (i=0; i<NFFT; i++) 
	{
		XXLS[i] = 0.;
		AALS[i] = 0.;
		EELS[i] = 0.;
	}
	
	for (i=0; i<N; i++)
	{
		k = (q + i - N/2);    // frequency bin associated with ith sample in GB data arrays
		f = (double)(k)/Tobs; // convert bin number to frequency
		
		iRe = 2*i;
		iIm = 2*i+1;
		
		XXLS[2*k]   += sqTobs*XX[iRe];
		XXLS[2*k+1] += sqTobs*XX[iIm];
		
		AALS[2*k]   += sqTobs*AA[iRe];
		AALS[2*k+1] += sqTobs*AA[iIm];
		
		EELS[2*k]   += sqTobs*EE[iRe];
		EELS[2*k+1] += sqTobs*EE[iIm];
	}
	
	
	

	fprintf(stdout, "\n==============================================================\n");
	
	// de-allocate memory
	free(params);
	free(XX);   free(AA);   free(EE);
	free(XXLS); free(AALS); free(EELS);
	
	return 0;
}

long get_N(double *params)
{
	// This determines the number of samples to take of the slowly evolving bit 
	// of the GB waveform. Right now only instrument noise is used in the estimate
	
	long mult, N, M;
	
	double amp, f0, fonfs;
	double SnX, SnAE;
	double Acut, Sm;
	
	f0  = params[0]/Tobs;
	amp = exp(params[3]);
	
	
	mult = 8;
	if((Tobs/YEAR) <= 8.0) mult = 8;
	if((Tobs/YEAR) <= 4.0) mult = 4;
	if((Tobs/YEAR) <= 2.0) mult = 2;
	if((Tobs/YEAR) <= 1.0) mult = 1;
	
	N = 32*mult;
	if(f0 > 0.001) N = 64*mult;
	if(f0 > 0.01)  N = 256*mult;
	if(f0 > 0.03)  N = 512*mult;
	if(f0 > 0.1)   N = 1024*mult;

	fonfs = f0/fstar;

	instrument_noise(f0, &SnAE, &SnX);

	//  calculate michelson noise 
	Sm = SnX/(4.0*sin(fonfs)*sin(fonfs));

	Acut = amp*sqrt(Tobs/Sm);

	M = (long)(pow(2.0,(rint(log(Acut)/log(2.0))+1.0)));

	if(M < N)    M = N;
	if(N < M)    N = M;
	if(M > 8192) M = 8192;

	N = M;

	return N;
}

