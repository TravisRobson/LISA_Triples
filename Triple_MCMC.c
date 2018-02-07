#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sf.h>
 
#include "Constants.h"
#include "GB.h"
#include "Triple.h"
#include "LISA.h"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage)
{
    double val = (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3.1f%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    
    fflush (stdout);
}

void scan_src(struct GB *gb, struct Triple *trip, char *argv[]);
void print_signal(struct GB *gb, FILE *fptr, double *XX, double *AA, double *EE);
void fill_data_stream(struct GB *gb, double *AA, double *EE, double *AALS, double *EELS);
void copy_gb(struct GB *dest, struct GB *src);
void calc_TayExp_Fisher_GR(struct GB *gb, double **Fisher);
void calc_TayExp_Fisher(struct GB *gb, double **Fisher);
void matrix_eigenstuff(double **matrix, double **evector, double *evalue, int N);
void prior_jump(struct GB *gb_y, gsl_rng *r, double f0);
void diff_ev_jump(struct GB *gb_x, struct GB *gb_y, gsl_rng *r, double ***history, long im, int c);
void Fisher_jump(struct GB *gb_y, struct GB *gb_x, gsl_rng *r, double **evecs, double *evals, double heat);
void check_priors(double *params, int *meets_priors, int NP, double f0, long N);

double get_mb(double Mchirp, double ma);
double get_snrAE(struct GB *gb, double *AA, double *EE);
double get_logL(struct GB *gb_x, struct GB *gb_y, double *AALS, double *EELS, double logL_cur);
double get_overlap(struct GB *gb, double *AA, double *EE, double *AALS, double *EELS);
double get_data_snr(double *AALS, double *EELS);
double invert_matrix(double **matrix, int N);

int main(int argc, char *argv[])
{		
 	int i, j, k, ic;
 	int NC;
 	int meets_priors, seed, hold, id;
 	int *who;

 	int *cnt, *acc, *swap_cnt, *swap_acc;
 		
 	long im;
 	const long NBURN = atoi(argv[7]);
	const long NMCMC = atoi(argv[8]);
	const long NCOOL = atoi(argv[11]);
 	
 	double snr_true, snr, snr_data, FF;
 	double cond, gamma, T_max, Ti;
 	double alpha, beta, loga, jump;
 	double logLy    = -1.0e30;
 	double logL_max = -1.0e30;
 	
 	double *XX, *AA, *EE;
 	double *AALS, *EELS;
 	double *logLx;
 	double **Fisher, **evecs, *evals;
 	double *heat, ***history;
 
 	FILE *out_file;
	
	fprintf(stdout, "==============================================================\n\n");
	
	fprintf(stdout, "Observation T: %.3f yrs\t df: %e mHz\n", Tobs/YEAR, 1.e3/Tobs);
	
	/* --------- Figure out the Temperature Ladder first --------- */
	T_max = atof(argv[5]);    // Target SNR first
	T_max = T_max*T_max*0.04; // Per suggestion of Assignment 3 (rho^{2}/25)
	i = 0;
	Ti = 1.;
	if (flag_GR == 0) gamma = 9.; // This is actually the dimension of the model
	else gamma = 8.;
	gamma = 1.2*(1. + sqrt(2./gamma));
	while (Ti < T_max)
	{
		Ti *= gamma;
		i++;
	}
	NC = i + 1;
	fprintf(stdout, "Suggested Temperature Ladder --> Gamma: %f\tNC: %d\n", gamma, NC);
	heat = malloc(NC*sizeof(double)); heat[0] = 1.;
	for (ic=1; ic<NC-1; ic++)
	{
		heat[ic]   = heat[ic-1]*gamma;
	}   heat[NC-1] = T_max;
	
	/* --------- Set up the ``true'' source --------- */
	struct GB *gb = malloc(sizeof(struct GB));
	struct Triple *trip = malloc(sizeof(struct Triple));
	scan_src(gb, trip, argv);
	
	/* --------- Calculate the true signal --------- */
	XX = malloc(2*gb->N*sizeof(double));
	AA = malloc(2*gb->N*sizeof(double));
	EE = malloc(2*gb->N*sizeof(double));
	
	GB_triple(gb, XX, AA, EE, trip);
	
	/* --------- Adjust signal amplitude to get appropriate SNR --------- */
	snr_true = get_snrAE(gb, AA, EE); 
	gb->params[3] = log(atof(argv[5])/snr_true*exp(gb->params[3]));
	GB_triple(gb, XX, AA, EE, trip);
	fprintf(stdout, "\n--------------------------------\n");
	snr_true = get_snrAE(gb, AA, EE); fprintf(stdout, "\nSNR: %f\n", snr_true);
	
	out_file = fopen(argv[4] ,"w");
	print_signal(gb, out_file, XX, AA, EE);
	
	/* --------- Construct full data-stream --------- */
	const long NFFT = (long)(Tobs/dt);
	AALS = malloc(NFFT*sizeof(double));
	EELS = malloc(NFFT*sizeof(double));
	fill_data_stream(gb, AA, EE, AALS, EELS);
	
	/* --------- array to hold current state of GBs --------- */
	struct GB **gb_x_arr = malloc(NC*sizeof(struct GB *));
	for (ic=0; ic<NC; ic++) gb_x_arr[ic] = malloc(sizeof(struct GB));
	for (ic=0; ic<NC; ic++)
	{
		if (flag_GR == 0) gb_x_arr[ic]->params = malloc((gb->NP+1)*sizeof(double));
		else gb_x_arr[ic]->params = malloc(gb->NP*sizeof(double));
	}
 	
	/* --------- setup the first guess at the GB parameters --------- */
	if (flag_GR == 0) gb_x_arr[0]->NP = 9; // i.e. f0, dfdt_0, d2fdt2_0 vary independently
	else gb_x_arr[0]->NP = 8;              // i.e. f0, dfdt_0 determine d2fdt2_0 
	gb_x_arr[0]->N  = pow(2, atoi(argv[2]));
	gb_x_arr[0]->T  = Tobs;
	gb_x_arr[0]->params[0] = gb->params[0];//*(1. + 0.*get_vLOS(trip, 0.)/C); // adjust to get closer to actual bin
	gb_x_arr[0]->params[1] = gb->params[1]; 
	gb_x_arr[0]->params[2] = gb->params[2];
	gb_x_arr[0]->params[3] = gb->params[3];
	gb_x_arr[0]->params[4] = gb->params[4];
	gb_x_arr[0]->params[5] = gb->params[5];
	gb_x_arr[0]->params[6] = gb->params[6];
	gb_x_arr[0]->params[7] = gb->params[7];
	if (gb_x_arr[0]->NP > 8) gb_x_arr[0]->params[8] = 11./3.*gb->params[7]*gb->params[7]/gb->params[0];
	gb_x_arr[0]->q  = (long)(gb_x_arr[0]->params[0]); 

	
	/* --------- Calculate the signal associated with the first guess --------- */
	if (flag_GR == 0) FAST_LISA(gb_x_arr[0]->params, gb_x_arr[0]->N, XX, AA, EE, gb_x_arr[0]->NP);
 	else			  FAST_LISA_GR(gb_x_arr[0]->params, gb_x_arr[0]->N, XX, AA, EE, gb_x_arr[0]->NP);
	snr = get_snrAE(gb_x_arr[0], AA, EE);
	gb_x_arr[0]->snr = snr;
	gb_x_arr[0]->overlap = get_overlap(gb_x_arr[0], AA, EE, AALS, EELS);
	
	/* --------- Calculate the Initial log-likelihood and overlap with true signal --------- */
	snr_data = get_data_snr(AALS, EELS);
	logLx = malloc(NC*sizeof(double));
	logLx[0] = -0.5*(snr_data*snr_data + snr*snr - 2.*gb_x_arr[0]->overlap);
	for (ic=1; ic<NC; ic++) logLx[ic] = logLx[0];
	fprintf(stdout, "logLx[0]: %f\n", logLx[0]);
	FF = gb_x_arr[0]->overlap/snr_true/snr; 
 	fprintf(stdout, "initial overlap: %f%%\n", FF*100.);	
	
	/* --------- Copy the other GBs --------- */
	for (ic=1; ic<NC; ic++) copy_gb(gb_x_arr[ic], gb_x_arr[0]);
 
	/* --------- Calculate Fisher Matrix and associated eigen-bs --------- */
	Fisher = malloc(gb_x_arr[0]->NP*sizeof(double *));
	for (i=0; i<gb_x_arr[0]->NP; i++) Fisher[i] = malloc(gb_x_arr[0]->NP*sizeof(double));
	
	for (i=0;i<gb_x_arr[0]->NP;i++)
	{
		for (j=0; j<gb_x_arr[0]->NP; j++) Fisher[i][j] = 0.;
	}
	
	if (flag_GR == 0) calc_TayExp_Fisher(gb_x_arr[0], Fisher);
	else calc_TayExp_Fisher_GR(gb_x_arr[0], Fisher);

	evecs = malloc(gb_x_arr[0]->NP*sizeof(double *));
	for (i=0; i<gb_x_arr[0]->NP; i++) evecs[i] = malloc(gb_x_arr[0]->NP*sizeof(double));
	evals = malloc(gb_x_arr[0]->NP*sizeof(double));
	for (i=0; i<gb_x_arr[0]->NP; i++) 
	{
		for (j=0; j<gb_x_arr[0]->NP; j++) evecs[i][j] = 0.;
		evals[i] = 0.;
	}
	
	cond = invert_matrix(Fisher, gb_x_arr[0]->NP); 
	if (cond > 6.) fprintf(stdout, "Condition number: %e!!!\n\n", cond);
 	
 	matrix_eigenstuff(Fisher, evecs, evals, gb_x_arr[0]->NP);	
 	
	/* --------- Setup proposal GB structure --------- */
	struct GB *gb_y = malloc(sizeof(struct GB));
	gb_y->params = malloc(gb_x_arr[0]->NP*sizeof(double));
	copy_gb(gb_y, gb_x_arr[0]);
 	
	/* --------- Setup chain identifier array --------- */
	who = malloc(NC*sizeof(int));
	for (ic=0; ic<NC; ic++) who[ic] = ic;
 		
	/* --------- Setup Posterior Max GB structure --------- */
	struct GB *gb_max = malloc(sizeof(struct GB));
    gb_max->params = malloc(gb_x_arr[0]->NP*sizeof(double));
    copy_gb(gb_max, gb_x_arr[0]);

	/* --------- Setup history tensor for burn-in --------- */
	history = malloc(NC*sizeof(double **));
	for (ic=0; ic<NC; ic++) history[ic] = malloc(NBURN*sizeof(double *));
	for (ic=0; ic<NC; ic++)
	{
		for (j=0; j<NBURN; j++) history[ic][j] = malloc(gb_x_arr[0]->NP*sizeof(double));
	}	
	
	/* --------- Keep track of acceptances --------- */
	swap_cnt = malloc((NC-1)*sizeof(int));
	swap_acc = malloc((NC-1)*sizeof(int));
	acc = malloc(NC*sizeof(int));
	cnt = malloc(NC*sizeof(int));
	for (ic=0; ic<NC-1; ic++)
	{	
		cnt[ic]      = 0;
		acc[ic]      = 0;
		swap_cnt[ic] = 0;
		swap_acc[ic] = 0;
	}
	cnt[NC-1] = 0; acc[NC-1] = 0;
 	
	/* --------- Setup random number generator --------- */
	seed  = atoi(argv[6]); 
	
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);
	gsl_rng_set(r, seed);
	
	gsl_rng **r_arr = malloc(NC*sizeof(gsl_rng *));
	for (ic=0; ic<NC; ic++) r_arr[ic] = gsl_rng_alloc(T);
	for (ic=0; ic<NC; ic++) gsl_rng_set(r_arr[ic], ic*seed);
	
	
	meets_priors = 1; // Priors are currently met
	fprintf(stdout, "Burning\n");
	/* --------- MCMC Burn In --------- */ 
	for (im=0; im<NBURN; im++) 
	{
		/* ---- Print Progress Bar ---- */
		if (im%(int)(0.01*(NBURN-1)) == 0. && NBURN > 100 && flag_prog == 1) 
		{
			printProgress((double)im/(double)(NBURN-1));
		}
		
		
		/* ---- Determine the type of proposal ---- */
		alpha = gsl_rng_uniform(r);
		
		/* ---- propose chain swap ---- */
		if (NC > 1 && alpha < 0.5)
		{	// Propose a jump between k^{th} and (k+1)^{th} chains
			alpha = (double)(NC-1)*gsl_rng_uniform(r);
			k = (int)(alpha); 
			swap_cnt[k]++;
			
			beta  = (logLx[who[k]] - logLx[who[k+1]])/heat[k+1];
			beta -= (logLx[who[k]] - logLx[who[k+1]])/heat[k];
		
			alpha = log(gsl_rng_uniform(r));

			if (beta > alpha)
			{
				hold     = who[k];
				who[k]   = who[k+1];
				who[k+1] = hold;
				swap_acc[k]++;
			}
		}	
		
		/* ---- propose normal MCMC update for each chain ---- */
		for (ic=0; ic<NC; ic++)
		{
			cnt[ic]++;    // proposed a jump, count it
			id = who[ic]; // determine the ID of the chain with (ic)^{th} temperature

			/* ---- Determine the type of proposal ---- */
			jump = gsl_rng_uniform(r);

			if (jump < 0.01) prior_jump(gb_y, r_arr[id], gb->params[0]/Tobs);
			else if (jump < 0.4 && jump > 0.01) Fisher_jump(gb_y, gb_x_arr[id], r_arr[id], evecs, evals, heat[ic]);
			else if (jump < 0.7 && jump > 0.5 && jump > 0.01) diff_ev_jump(gb_x_arr[id], gb_y, r_arr[id], history, im, ic);
			else 
			{	// propose to jump the source one bin
				alpha = gsl_rng_uniform(r);
				if (alpha < 0.5) gb_y->params[0] += 1.;
				else 			 gb_y->params[0] -= 1.;
				gb_y->q = (long)(gb_y->params[0]);
			}
			check_priors(gb_y->params, &meets_priors, gb_y->NP, gb->params[0]/Tobs, gb->N);

			/* ---- Make a decision ---- */
			if (meets_priors == 1)
			{
				if (flag_GR == 0) FAST_LISA(gb_y->params, gb_y->N, XX, AA, EE, gb_y->NP); 
				else		      FAST_LISA_GR(gb_y->params, gb_y->N, XX, AA, EE, gb_y->NP); 
				
				gb_y->snr     = get_snrAE(gb_y, AA, EE);
				gb_y->overlap = get_overlap(gb_y, AA, EE, AALS, EELS);
				logLy = get_logL(gb_x_arr[id], gb_y, AALS, EELS, logLx[id]);					
				loga  = log(gsl_rng_uniform(r));

				if (logLy > -INFINITY && loga < (logLy - logLx[id])/heat[ic])
				{
					acc[ic]++;    // accepted a jump, count it

					copy_gb(gb_x_arr[id], gb_y);
					logLx[id] = logLy;
				
					/* ---- store the max likelihood values ---- */
					if (ic == 0 && logLx[who[0]] > logL_max)
					{   // only for cold chain
						logL_max   = logLx[who[0]];
						copy_gb(gb_max, gb_x_arr[who[0]]);
					}
				}
			}
		}
			
		/* ---- Update history tensor ---- */
		for (ic=0; ic<NC; ic++)
		{
			for (j=0; j<gb_y->NP; j++) history[ic][im][j] = gb_x_arr[who[ic]]->params[j]; 
		}
		
		/* ---- Update Fisher Matrix ---- */
		if (im%100 == 0)
		{
			if (flag_GR == 0) calc_TayExp_Fisher(gb_max, Fisher);
			else 			  calc_TayExp_Fisher_GR(gb_max, Fisher);
			
			//invert_matrix(Fisher, gb_max->NP);
			matrix_eigenstuff(Fisher, evecs, evals, gb_max->NP);
		}
	}
	printProgress(1.);
	for (ic=0; ic<NC; ic++)
	{
		for (j=0; j<NBURN; j++) free(history[ic][j]);
		free(history[ic]);
	}	free(history);
	
	/* ---- Burn in data ---- */
	fprintf(stdout, "\n\n"); fprintf(stdout, "Burn In Summary\n--------------------------------\n");
	
	for (ic=0; ic<NC-1; ic++) 
	{
		fprintf(stdout, "logL[%d]: %f\t", ic, logLx[who[ic]]);
		if (cnt[ic] > 0) fprintf(stdout, "Accp Rate[%d]: %f%%\t", ic, (double)acc[ic]/(double)cnt[ic]*100.);
		if (swap_cnt[ic] > 0) fprintf(stdout, "Swap Rate[%d<-->%d]: %f%%\n", ic, ic+1, (double)swap_acc[ic]/(double)swap_cnt[ic]*100.);
	}
	fprintf(stdout, "logL[%d]: %f\t", NC-1, logLx[who[NC-1]]);
	if (cnt[ic] > 0) fprintf(stdout, "Accp Rate[%d]: %f%%\n", NC-1, (double)acc[NC-1]/(double)cnt[NC-1]*100.);
	
	fprintf(stdout, "\nMax logP: %f\n", logL_max);
	FF = gb_max->overlap/snr_true/gb_max->snr; 
 	fprintf(stdout, "FF: %f%%\n", FF*100.);	
 	 	
 	/* ---- Reset history tensor ---- */
 	history = malloc(NC*sizeof(double **));
	for (ic=0; ic<NC; ic++) history[ic] = malloc(NMCMC*sizeof(double *));
	for (ic=0; ic<NC; ic++)
	{
		for (j=0; j<NMCMC; j++) history[ic][j] = malloc(gb_x_arr[0]->NP*sizeof(double));
	}	
 	
 	fprintf(stdout, "\nMCMC\n"); out_file = fopen(argv[9] ,"w");
	/* --------- MCMC Burn In --------- */ 
	for (im=0; im<NMCMC; im++) 
	{
		/* ---- Print Progress Bar ---- */
		if (im%(int)(0.01*(NMCMC-1)) == 0. && NMCMC > 100 && flag_prog == 1) 
		{
			printProgress((double)im/(double)(NMCMC-1));
		}
		
		
		/* ---- Determine the type of proposal ---- */
		alpha = gsl_rng_uniform(r);
		
		/* ---- propose chain swap ---- */
		if (NC > 1 && alpha < 0.5)
		{	// Propose a jump between k^{th} and (k+1)^{th} chains
			alpha = (double)(NC-1)*gsl_rng_uniform(r);
			k = (int)(alpha); 
			swap_cnt[k]++;
			
			beta  = (logLx[who[k]] - logLx[who[k+1]])/heat[k+1];
			beta -= (logLx[who[k]] - logLx[who[k+1]])/heat[k];
		
			alpha = log(gsl_rng_uniform(r));

			if (beta > alpha)
			{
				hold     = who[k];
				who[k]   = who[k+1];
				who[k+1] = hold;
				swap_acc[k]++;
			}
		}	
		
		/* ---- propose normal MCMC update for each chain ---- */
		for (ic=0; ic<NC; ic++)
		{
			cnt[ic]++;    // proposed a jump, count it
			id = who[ic]; // determine the ID of the chain with (ic)^{th} temperature

			/* ---- Determine the type of proposal ---- */
			jump = gsl_rng_uniform(r);

			if (jump < 0.01) prior_jump(gb_y, r_arr[id], gb->params[0]/Tobs);
			else if (jump < 0.4 && jump > 0.01) Fisher_jump(gb_y, gb_x_arr[id], r_arr[id], evecs, evals, heat[ic]);
			else if (jump < 0.7 && jump > 0.5 && jump > 0.01) diff_ev_jump(gb_x_arr[id], gb_y, r_arr[id], history, im, ic);
			else 
			{	// propose to jump the source one bin
				alpha = gsl_rng_uniform(r);
				if (alpha < 0.5) gb_y->params[0] += 1.;
				else 			 gb_y->params[0] -= 1.;
				gb_y->q = (long)(gb_y->params[0]);
			}
			check_priors(gb_y->params, &meets_priors, gb_y->NP, gb->params[0]/Tobs, gb->N);

			/* ---- Make a decision ---- */
			if (meets_priors == 1)
			{
				if (flag_GR == 0) FAST_LISA(gb_y->params, gb_y->N, XX, AA, EE, gb_y->NP); 
				else		      FAST_LISA_GR(gb_y->params, gb_y->N, XX, AA, EE, gb_y->NP); 
				
				gb_y->snr     = get_snrAE(gb_y, AA, EE);
				gb_y->overlap = get_overlap(gb_y, AA, EE, AALS, EELS);
				logLy = get_logL(gb_x_arr[id], gb_y, AALS, EELS, logLx[id]);					
				loga  = log(gsl_rng_uniform(r));

				if (logLy > -INFINITY && loga < (logLy - logLx[id])/heat[ic])
				{
					acc[ic]++;    // accepted a jump, count it

					copy_gb(gb_x_arr[id], gb_y);
					logLx[id] = logLy;
				
					/* ---- store the max likelihood values ---- */
					if (ic == 0 && logLx[who[0]] > logL_max)
					{   // only for cold chain
						logL_max   = logLx[who[0]];
						copy_gb(gb_max, gb_x_arr[who[0]]);
					}
				}
			}
		}
			
		/* ---- Update history tensor ---- */
		for (ic=0; ic<NC; ic++)
		{
			for (j=0; j<gb_y->NP; j++) history[ic][im][j] = gb_x_arr[who[ic]]->params[j]; 
		}
		
		/* ---- Update Fisher Matrix ---- */
		if (im%100 == 0)
		{
			if (flag_GR == 0) calc_TayExp_Fisher(gb_max, Fisher);
			else 			  calc_TayExp_Fisher_GR(gb_max, Fisher);
			
			//invert_matrix(Fisher, gb_max->NP);
			matrix_eigenstuff(Fisher, evecs, evals, gb_max->NP);
		}
		
		/* ---- Print to chain file ---- */
		if (im%10 == 0)
		{
			fprintf(out_file, "%ld ", im/10);
			fprintf(out_file, "%.12g ", logLx[who[0]]);
			for (j=0; j<gb_y->NP; j++) fprintf(out_file, "%.12g ", gb_x_arr[who[0]]->params[j]);
			fprintf(out_file, "\n");
		}
		
	}
	printProgress(1.);
	/* ---- MCMC data ---- */
	fprintf(stdout, "\n\n"); fprintf(stdout, "MCMC Summary\n--------------------------------\n");
	
	for (ic=0; ic<NC-1; ic++) 
	{
		fprintf(stdout, "logL[%d]: %f\t", ic, logLx[who[ic]]);
		if (cnt[ic] > 0) fprintf(stdout, "Accp Rate[%d]: %f%%\t", ic, (double)acc[ic]/(double)cnt[ic]*100.);
		if (swap_cnt[ic] > 0) fprintf(stdout, "Swap Rate[%d<-->%d]: %f%%\n", ic, ic+1, (double)swap_acc[ic]/(double)swap_cnt[ic]*100.);
	}
	fprintf(stdout, "logL[%d]: %f\t", NC-1, logLx[who[NC-1]]);
	if (cnt[ic] > 0) fprintf(stdout, "Accp Rate[%d]: %f%%\n", NC-1, (double)acc[NC-1]/(double)cnt[NC-1]*100.);
	
	fprintf(stdout, "\nMax logP: %f\n", logL_max);
	FF = gb_max->overlap/snr_true/gb_max->snr; 
 	fprintf(stdout, "FF: %f%%\n", FF*100.);		
	
	/* ---- Print Max Posterior Source ---- */
	if (flag_GR == 0) FAST_LISA(gb_max->params, gb_max->N, XX, AA, EE, gb_max->NP); 
	else		      FAST_LISA_GR(gb_max->params, gb_max->N, XX, AA, EE, gb_max->NP); 
	out_file = fopen(argv[10] ,"w");
	print_signal(gb_max, out_file, XX, AA, EE);
	
	
	
	/* ---- Cool Down to find peak ---- */
	fprintf(stdout, "\nCool Down\n");
	for (im=0; im<NCOOL; im++) 
	{
		/* ---- Adjust Temperature Ladder for Simulated Annealing ---- */
		heat[0]    = pow(10.0, -3.*(double)(im)/(double)(NCOOL)); 
		for (ic=1; ic<NC; ic++) heat[ic] = gamma*heat[ic-1];
		
		/* ---- Print Progress Bar ---- */
		if (im%(int)(0.01*(NCOOL-1)) == 0. && NCOOL > 100 && flag_prog == 1) 
		{
			printProgress((double)im/(double)(NCOOL-1));
		}
		
		/* ---- Determine the type of proposal ---- */
		alpha = gsl_rng_uniform(r);
		
		/* ---- propose chain swap ---- */
		if (NC > 1 && alpha < 0.5)
		{	// Propose a jump between k^{th} and (k+1)^{th} chains
			alpha = (double)(NC-1)*gsl_rng_uniform(r);
			k = (int)(alpha); 
			
			beta  = (logLx[who[k]] - logLx[who[k+1]])/heat[k+1];
			beta -= (logLx[who[k]] - logLx[who[k+1]])/heat[k];
		
			alpha = log(gsl_rng_uniform(r));

			if (beta > alpha)
			{
				hold     = who[k];
				who[k]   = who[k+1];
				who[k+1] = hold;
			}
		}	
		
		/* ---- propose normal MCMC update for each chain ---- */
		for (ic=0; ic<NC; ic++)
		{
			id = who[ic]; // determine the ID of the chain with (ic)^{th} temperature

			/* ---- Determine the type of proposal ---- */
			jump = gsl_rng_uniform(r);

			if (jump < 0.01) prior_jump(gb_y, r_arr[id], gb->params[0]/Tobs);
			else if (jump < 0.4 && jump > 0.01) Fisher_jump(gb_y, gb_x_arr[id], r_arr[id], evecs, evals, heat[ic]);
			else if (jump < 0.7 && jump > 0.5 && jump > 0.01) diff_ev_jump(gb_x_arr[id], gb_y, r_arr[id], history, NMCMC-1, ic);
			else 
			{	// propose to jump the source one bin
				alpha = gsl_rng_uniform(r);
				if (alpha < 0.5) gb_y->params[0] += 1.;
				else 			 gb_y->params[0] -= 1.;
				gb_y->q = (long)(gb_y->params[0]);
			}
			check_priors(gb_y->params, &meets_priors, gb_y->NP, gb->params[0]/Tobs, gb->N);

			/* ---- Make a decision ---- */
			if (meets_priors == 1)
			{
				if (flag_GR == 0) FAST_LISA(gb_y->params, gb_y->N, XX, AA, EE, gb_y->NP); 
				else		      FAST_LISA_GR(gb_y->params, gb_y->N, XX, AA, EE, gb_y->NP); 
				
				gb_y->snr     = get_snrAE(gb_y, AA, EE);
				gb_y->overlap = get_overlap(gb_y, AA, EE, AALS, EELS);
				logLy = get_logL(gb_x_arr[id], gb_y, AALS, EELS, logLx[id]);					
				loga  = log(gsl_rng_uniform(r));

				if (logLy > -INFINITY && loga < (logLy - logLx[id])/heat[ic])
				{
					copy_gb(gb_x_arr[id], gb_y);
					logLx[id] = logLy;
				
					/* ---- store the max likelihood values ---- */
					if (ic == 0 && logLx[who[0]] > logL_max)
					{   // only for cold chain
						logL_max   = logLx[who[0]];
						copy_gb(gb_max, gb_x_arr[who[0]]);
					}
				}
			}
		}
	}
	printProgress(1.);
	for (ic=0; ic<NC; ic++)
	{
		for (j=0; j<NMCMC; j++) free(history[ic][j]);
		free(history[ic]);
	}	free(history);
	
	fprintf(stdout, "\nMax logP: %f\n", logL_max);
	FF = gb_max->overlap/snr_true/gb_max->snr; 
 	fprintf(stdout, "FF: %f%%\n", FF*100.);		
	
	/* ---- Print Max Posterior Source ---- */
	if (flag_GR == 0) FAST_LISA(gb_max->params, gb_max->N, XX, AA, EE, gb_max->NP); 
	else		      FAST_LISA_GR(gb_max->params, gb_max->N, XX, AA, EE, gb_max->NP); 
	out_file = fopen(argv[10] ,"w");
	print_signal(gb_max, out_file, XX, AA, EE);
	
	
	
 	
 	fprintf(stdout, "\n==============================================================\n");
	
	free(logLx);

 	free(cnt); free(acc); free(swap_cnt); free(swap_acc);
 	
	for(i=0; i<gb_x_arr[0]->NP; i++) 
	{
		free(Fisher[i]);
		free(evecs[i]);
	}
	free(evals); free(evecs); free(Fisher);
	
	free(gb->params);
	free(gb);
	
	free(trip->params);
	free(trip);
	
	free(XX); free(AA); free(EE);
 	free(AALS); free(EELS);
 	
	for (ic=0; ic<NC; ic++)
	{
		free(gb_x_arr[ic]->params);
		free(gb_x_arr[ic]);
	}	free(gb_x_arr);
 	
 	free(heat);
 	free(who);
 	
	free(gb_max->params);
	free(gb_max);
 	
	free(gb_y->params);
	free(gb_y);
 	
	gsl_rng_free(r);
	for (ic=0; ic<NC; ic++) gsl_rng_free(r_arr[ic]);
	free(r_arr);
	
	return 0;
}

void scan_src(struct GB *gb, struct Triple *trip, char *argv[])
{
	// GB parameters
	double f0, dfdt_0, co_lat, phi, amp, iota, psi, phi0; 
	
	// The rest of the parameters
	double ma, mb, mc, Mc, m2;
	double e2, P2, n2, a2;
	double Ac1c2, v_on_c;
	
	// Stability Criteria (Eggleton & Kiseleva 1995)
	double q_1, q_2, Y_EK, Y_MA, r_ap, a1;
	
	FILE *in_file;
	
	in_file = fopen(argv[1], "r");
	fscanf(in_file, "%lg",   &f0);
	fscanf(in_file, "%lg",   &dfdt_0);
	fscanf(in_file, "%lg",   &co_lat);
	fscanf(in_file, "%lg",   &phi);
	fscanf(in_file, "%lg",   &amp);
	fscanf(in_file, "%lg",   &iota);
	fscanf(in_file, "%lg",   &psi);
	fscanf(in_file, "%lg",   &phi0);
	
	gb->N  = pow(2, atoi(argv[2]));
	gb->T  = Tobs;
	gb->q  = (long)(f0*gb->T);
  
	gb->NP = 8; // This deals with the source frame GB (i.e. d2fdt2_0 is determined by GR)
	
	gb->params = malloc(gb->NP*sizeof(double));
	
	// set the parameters
	gb->params[0] = f0*Tobs;
	gb->params[1] = cos(PIon2 - co_lat); // convert to spherical polar
	gb->params[2] = phi;
	gb->params[3] = log(amp);
	gb->params[4] = cos(iota);
	gb->params[5] = psi;
	gb->params[6] = phi0;
	gb->params[7] = dfdt_0*Tobs*Tobs;
	
	fscanf(in_file, "\n"); // go to next line in source file to read off triples parameters
	fscanf(in_file, "%lg", &ma); // (Solar mass)
	fscanf(in_file, "%lg", &mc); // (Solar mass)
	fscanf(in_file, "%lg", &Mc); // (Solar mass)
	fscanf(in_file, "%lg", &e2);
	fscanf(in_file, "%lg", &P2); // in years
	fclose(in_file);
	
	P2   *= YEAR;          			     // adjust to (sec)
	mb    = get_mb(Mc, ma);  			 // mass of second body in tight binary
	m2    = ma + mb + mc;   		     // total mass of triple	
	n2    = PI2/P2;		  			     // outer orbit mean motion
	a2    = pow(G*m2*MSUN/n2/n2, 1./3.); // semi-major axis of outer orbit
	Ac1c2 = 0.77; 						 // This is the average value	

 	trip->NP = 5;	//  amp2, omegabar, e2, n2, T2
 	
 	trip->params = malloc(trip->NP*sizeof(double));
 	
	trip->params[0] = log(mc*MSUN*sqrt(G/(m2*MSUN)/a2/(1. - e2*e2))*Ac1c2);
	trip->params[1] = 0.; 
	trip->params[2] = e2; 
	trip->params[3] = log(n2*YEAR);
	trip->params[4] = log(atof(argv[3])*P2/YEAR);
	
	v_on_c = mc*MSUN*sqrt(G/(m2*MSUN)/a2/(1. - e2*e2))*Ac1c2/C;
	
	// Refer to Eggleton & Kiseleva 1995, Mardling & Aarseth 2001, He & Petrovich 2018
	if (ma > mb) q_1 = ma/mb;
	else q_1 = mb/ma;
	q_2 = (ma+mb)/mc;
	Y_EK = 1. + 3.7/pow(q_2, 1./3.) + 2.2/(1. + pow(q_2, 1./3.)) + 1.4/pow(q_1, 1./3.)*(pow(q_2, 1./3.)-1.)/(pow(q_2, 1./3.)+1.);
	
	a1 = pow(G*MSUN*(ma+mb)/(M_PI*f0)/(M_PI*f0), 1./3.);
	
	r_ap = a2*(1. - e2)/(a1*(1. - 0.)); // 0. -> e1
	
	Y_MA = 2.8/(1. - 0.)*pow((1. + mc/(ma+mb))*(1. + e2)/sqrt(1. - e2), 2./5.); // 0.->e1, Excluded the (1. - 0.3 i_{m}/180 deg) factor
	
	fprintf(stdout, "\nInner Orbit\n");
	fprintf(stdout, "----------------------------------\n");
	fprintf(stdout, "P_{1,0} (f_{0, GW}): %f min (%f mHz)\n", 2./f0/60. ,f0*1.0e3);
	fprintf(stdout, "m_{a}: %f MSUN,\tm_{b}: %f MSUN\n", ma, mb);
	fprintf(stdout, "m_{1}: %f MSUN,\tMc: %f MSUN\n", ma+mb, Mc);
	fprintf(stdout, "e_{1}: %f,\ta_{1,0}: %e km\n", 0., a1*1.0e-3);
	fprintf(stdout, "LISA Doppler 2f_{0}v/c*sin(theta): %e (%f) mHz (bins)\n", 0.2*f0*sin(PIon2 - co_lat), 0.2*f0*sin(PIon2 - co_lat)*Tobs/1000.);
	fprintf(stdout, "dfdt_{0}: %e (bins)\n", dfdt_0*Tobs*Tobs);
	fprintf(stdout, "GR d2fdt2_{0}: %e (bins)\n", 11./3.*dfdt_0*dfdt_0/f0*Tobs*Tobs*Tobs);
		
	fprintf(stdout, "\nOuter Orbit\n");
	fprintf(stdout, "----------------------------------\n");
	fprintf(stdout, "P_{2} (f_{2, orb}): %f min (%e mHz)\n", P2/60. , 1./P2*1.0e3);
	fprintf(stdout, "f0(v_{LOS}/c): %f (%f) mHz (bins)\n", f0*v_on_c*1.e3, Tobs*f0*v_on_c);
	fprintf(stdout, "m_{c}: %f MSUN,\tm_{2}: %f MSUN\n", mc, ma+mb+mc);
 	fprintf(stdout, "e_{2}: %f,\ta_{1,0}: %e km\n", e2, a2*1.0e-3);
	
	fprintf(stdout, "\nStability Criteria \n");
	fprintf(stdout, "----------------------------------\n");
	fprintf(stdout, "(EK: Eggleton & Kiseleva '95) and (MA: Mardling & Aarseth '01)\n");
	fprintf(stdout, "Y_{EK}: %f\tY_{MA}: %f\n", Y_EK, Y_MA);
	fprintf(stdout, "r_{ap}: %f", r_ap);	
 	if (r_ap > Y_EK && r_ap > Y_MA) fprintf(stdout, "\tStability Satisfied!\n");
 	else fprintf(stdout, "\tUnstable Orbits!\n");
	
	return;
}

double get_mb(double Mchirp, double ma)
{
	double mb;
	double a;
	
	a = pow(9.*pow(ma, 5./2.) + sqrt(81.*ma*ma*ma*ma*ma 
								- 12.*Mchirp*Mchirp*Mchirp*Mchirp*Mchirp), 1./3.);
	
	mb  = pow(Mchirp, 5./3.)*(2.*pow(3., 1./3.)*pow(Mchirp, 5./3.) + pow(2., 1./3.)*a*a);
	
	mb /= pow(6., 2./3.)*pow(ma, 3./2.)*a;
	
	return mb;
}	

void print_signal(struct GB *gb, FILE *fptr, double *XX, double *AA, double *EE)
{
	int i, iRe, iIm;
	const int N = gb->N;
	
	long k;
	
	double f, sqT;
	
	sqT = sqrt(Tobs);
	
	for (i=0; i<N; i++)
	{
		k = (gb->q + i - N/2); // frequency bin associated with ith sample in GB data arrays
		f = (double)(k)/Tobs;  // convert bin number to frequency
		
		iRe = 2*i;
		iIm = 2*i+1;
		
		fprintf(fptr, "%.12g %e %e %e %e %e %e\n", f,  sqT*XX[iRe], sqT*XX[iIm],
													   sqT*AA[iRe], sqT*AA[iIm],
													   sqT*EE[iRe], sqT*EE[iIm]);
	}
	
	fclose(fptr);

	return;
}

double get_snrAE(struct GB *gb, double *AA, double *EE)
{
	int i, iRe, iIm;
	
	long k;
	
	const int N = gb->N;
	
	double f, SnAE, SnX;
	double snr = 0.;
	
	for (i=0; i<N; i++)
	{
		k = (gb->q + i - N/2);    
		f = (double)(k)/Tobs; 			
		instrument_noise(f, &SnAE, &SnX);
		
		iRe = 2*i;		
		iIm = 2*i+1;   
		
		snr += (AA[iRe]*AA[iRe] + AA[iIm]*AA[iIm] + EE[iRe]*EE[iRe] + EE[iIm]*EE[iIm])/SnAE;
	}
	snr *= 4.0*gb->T;
	snr  = sqrt(snr); 
	
	return snr;
}
 
void fill_data_stream(struct GB *gb, double *AA, double *EE, double *AALS, double *EELS)
{
	const int N = gb->N;
	
	int i, iRe, iIm;
	long k, kRe, kIm;
	
	const long NFFT = (long)(Tobs/dt);
	
	double sqT = sqrt(Tobs);
		
	for (i=0; i<NFFT; i++) 
	{
		AALS[i] = 0.;
		EELS[i] = 0.;
	}
	
	for (i=0; i<N; i++)
	{
		k = (gb->q + i - N/2);   // frequency bin associated with ith sample in GB data arrays
		
		iRe = 2*i;	 kRe = 2*k;
		iIm = 2*i+1; kIm = 2*k+1;
		
		AALS[kRe] += sqT*AA[iRe];
		AALS[kIm] += sqT*AA[iIm];
		
		EELS[kRe] += sqT*EE[iRe];
		EELS[kIm] += sqT*EE[iIm];
	}
	
	return;
}

void copy_gb(struct GB *dest, struct GB *src)
{
	int i;
	
	const int NP = src->NP;
	
	dest->N  = src->N;
	dest->T  = Tobs;
	dest->NP = src->NP;    
	dest->q  = src->q;
	
	for (i=0; i<NP; i++) dest->params[i] = src->params[i];
	
	dest->snr = src->snr;
	dest->overlap = src->overlap;
	
	return;
}

double get_data_snr(double *AALS, double *EELS)
{
	long i, iRe, iIm;
	
	double f, SnAE, SnX;
	
	double snr_data = 0.; 
	for (i=(long)(0.1*1.0e-3*Tobs); i<(long)(20.*1.0e-3*Tobs); i++)
	{
		f = (double)(i)/Tobs; 			
		instrument_noise(f, &SnAE, &SnX);
		
		iRe = 2*i; iIm = 2*i+1;
		
		snr_data += (AALS[iRe]*AALS[iRe] + AALS[iIm]*AALS[iIm] + EELS[iRe]*EELS[iRe] + EELS[iIm]*EELS[iIm])/SnAE;
	}
		
	return sqrt(4*snr_data);
}

double get_logL(struct GB *gb_x, struct GB *gb_y, double *AALS, double *EELS, double logL_cur)
{
	int flag = 0; // for checking detailed balance
	double logL;
	
	if (flag == 0)
	{
// 		double *XX, *AA, *EE;
// 		double *xx, *aa, *ee;
		double snr_old, snr_new;
		double overlap_old, overlap_new;
		
// 		XX = malloc(2*gb_y->N*sizeof(double)); xx = malloc(2*gb_x->N*sizeof(double));
// 		AA = malloc(2*gb_y->N*sizeof(double)); aa = malloc(2*gb_x->N*sizeof(double));
// 		EE = malloc(2*gb_y->N*sizeof(double)); ee = malloc(2*gb_x->N*sizeof(double));
		
		// Calculate old signal
// 		if (flag_GR == 0) FAST_LISA(gb_x->params, gb_x->N, xx, aa, ee, gb_x->NP);
//  		else FAST_LISA_GR(gb_x->params, gb_x->N, xx, aa, ee, gb_x->NP);
//  		
//  		// Calculate new signal
//  		if (flag_GR == 0) FAST_LISA(gb_y->params, gb_y->N, XX, AA, EE, gb_y->NP);
//  		else FAST_LISA_GR(gb_y->params, gb_y->N, XX, AA, EE, gb_y->NP);
		
		snr_new = gb_y->snr; //get_snrAE(gb_y, AA, EE);
		snr_old = gb_x->snr; 
		
		overlap_new = gb_y->overlap; //get_overlap(gb_y, AA, EE, AALS, EELS);
		overlap_old = gb_x->overlap; 
		
		logL  = logL_cur + 0.5*snr_old*snr_old - overlap_old;
		logL +=          - 0.5*snr_new*snr_new + overlap_new;
		
// 		free(XX); free(xx);
// 		free(AA); free(aa);
// 		free(EE); free(ee);
	}
	else logL = -1.;
	
	return logL;
}

double get_overlap(struct GB *gb, double *AA, double *EE, double *AALS, double *EELS)
{
	const int N = gb->N;
	int i, iRe, iIm; 

	long k, kRe, kIm;
	
	double overlap = 0.;
	double f, SnAE, SnX;
	
	for (i=0; i<N; i++)
	{
		k = (gb->q + i - N/2);    
		f = (double)(k)/Tobs; 			
		instrument_noise(f, &SnAE, &SnX);
		
		iRe = 2*i;	  kRe = 2*k;
		iIm = 2*i+1;  kIm = 2*k+1;  
		
		overlap += (AA[iRe]*AALS[kRe] + AA[iIm]*AALS[kIm] + EE[iRe]*EELS[kRe] + EE[iIm]*EELS[kIm])/SnAE;
	}
	overlap *= 4.0*sqrt(Tobs);
	
	return overlap;
}

void calc_TayExp_Fisher(struct GB *gb, double **Fisher)
{
	int i, j, k, m;
	int iRe, iIm;
	
	const int NP = gb->NP;
	const int N  = gb->N;
	
	const double ep = 1.0e-4;
	double arg, f;
	double SnAE, SnX;
	
	double *XX_p, *AA_p, *EE_p;
	double *XX_m, *AA_m, *EE_m;
	double **XX_d, **AA_d, **EE_d;
	
	struct GB *gb_p = malloc(sizeof(struct GB));
	struct GB *gb_m = malloc(sizeof(struct GB));
	
	gb_p->N 	 = N; //pow(2, p);
	gb_m->N 	 = N; //pow(2, p);
	gb_p->T 	 = Tobs;
	gb_m->T 	 = Tobs;
	gb_p->NP     = NP;
	gb_m->NP	 = NP;
	gb_p->params = malloc(NP*sizeof(double));
	gb_m->params = malloc(NP*sizeof(double));			
	gb_p->q      = (long)(gb->params[0]); // the bin really won't change
	gb_m->q      = (long)(gb->params[0]);
	
	XX_p = malloc(2*N*sizeof(double));
	AA_p = malloc(2*N*sizeof(double));
	EE_p = malloc(2*N*sizeof(double));
	XX_m = malloc(2*N*sizeof(double));
	AA_m = malloc(2*N*sizeof(double));
	EE_m = malloc(2*N*sizeof(double));
	
	XX_d = malloc(gb->NP*sizeof(double *));
	for (i=0; i<gb->NP; i++) XX_d[i] = malloc(2*gb->N*sizeof(double));
	AA_d = malloc(gb->NP*sizeof(double *));
	for (i=0; i<gb->NP; i++) AA_d[i] = malloc(2*gb->N*sizeof(double));
	EE_d = malloc(gb->NP*sizeof(double *));
	for (i=0; i<gb->NP; i++) EE_d[i] = malloc(2*gb->N*sizeof(double));
	
	for (i=0; i<2*N; i++)
	{
		XX_p[i] = 0.;
		AA_p[i] = 0.;
		EE_p[i] = 0.;
		XX_m[i] = 0.;
		AA_m[i] = 0.;
		EE_m[i] = 0.;
	}
	
	for (i=0; i<NP; i++)
	{
		gb_p->params[i] = 0.;
		gb_m->params[i] = 0.;
	}
	
	for (i=0; i<gb->NP; i++) 
	{
		for(j=0; j<2*gb->N; j++) 
		{
			XX_d[i][j] = 0.;
			AA_d[i][j] = 0.;
			EE_d[i][j] = 0.;
		}
	}
	
	// calculate the derivatives as a function of frequency
	for (i=0; i<NP; i++)
	{
		for (j=0; j<NP; j++)
		{
			gb_p->params[j] = gb->params[j];
			gb_m->params[j] = gb->params[j];
		}

		gb_p->params[i] += ep;
		gb_m->params[i] -= ep;
		
		FAST_LISA(gb_p->params, gb_p->N, XX_p, AA_p, EE_p, gb_p->NP);
		FAST_LISA(gb_m->params, gb_m->N, XX_m, AA_m, EE_m, gb_m->NP);
		
		for (j=0; j<2*N; j++) 
		{
			XX_d[i][j] = (XX_p[j] - XX_m[j])/(2.0*ep);
			AA_d[i][j] = (AA_p[j] - AA_m[j])/(2.0*ep);
			EE_d[i][j] = (EE_p[j] - EE_m[j])/(2.0*ep);
		}
	}
	free(XX_p); free(AA_p); free(EE_p);
	free(XX_m); free(AA_m); free(EE_m);
	free(gb_p->params); free(gb_m->params);
	free(gb_p); free(gb_m);
	
	
	// NWIP the derivative arrays
	for (i=0; i<NP; i++)
	{
		for(j=i; j<NP; j++)
		{
			arg = 0.;
			for (m=0; m<N; m++)
			{
				k = (gb->q + m - gb->N/2);    
				f = (double)(k)/Tobs; 			
				instrument_noise(f, &SnAE, &SnX);
		
				iRe = 2*m;		
				iIm = 2*m+1;   
		
				arg +=  (AA_d[i][iRe]*AA_d[j][iRe] + AA_d[i][iIm]*AA_d[j][iIm] 
					    +EE_d[i][iRe]*EE_d[j][iRe] + EE_d[i][iIm]*EE_d[j][iIm])/SnAE;
			}
			arg *= 4.*Tobs;
			
			Fisher[i][j] = arg;
		}
	}
	for(i=0; i<NP; i++)
	{
		free(XX_d[i]);
		free(AA_d[i]);
		free(EE_d[i]);
	}  
	free(XX_d); free(AA_d); free(EE_d);
	
	// make use of symmetry
	for (i=0; i<NP; i++)
	{
		for (j=i+1; j<NP; j++)
		{
			Fisher[j][i] = Fisher[i][j];
		}
	}
	return;
}

void calc_TayExp_Fisher_GR(struct GB *gb, double **Fisher)
{
	int i, j, k, m;
	int iRe, iIm;
	
	const int NP = gb->NP;
	const int N  = gb->N;
	
	const double ep = 1.0e-4;
	double arg, f;
	double SnAE, SnX;
	
	double *XX_p, *AA_p, *EE_p;
	double *XX_m, *AA_m, *EE_m;
	double **XX_d, **AA_d, **EE_d;
	
	struct GB *gb_p = malloc(sizeof(struct GB));
	struct GB *gb_m = malloc(sizeof(struct GB));
	
	gb_p->N 	 = N; //pow(2, p);
	gb_m->N 	 = N; //pow(2, p);
	gb_p->T 	 = Tobs;
	gb_m->T 	 = Tobs;
	gb_p->NP     = NP;
	gb_m->NP	 = NP;
	gb_p->params = malloc(NP*sizeof(double));
	gb_m->params = malloc(NP*sizeof(double));			
	gb_p->q      = (long)(gb->params[0]); // the bin really won't change
	gb_m->q      = (long)(gb->params[0]);
	
	XX_p = malloc(2*N*sizeof(double));
	AA_p = malloc(2*N*sizeof(double));
	EE_p = malloc(2*N*sizeof(double));
	XX_m = malloc(2*N*sizeof(double));
	AA_m = malloc(2*N*sizeof(double));
	EE_m = malloc(2*N*sizeof(double));
	
	XX_d = malloc(gb->NP*sizeof(double *));
	for (i=0; i<gb->NP; i++) XX_d[i] = malloc(2*gb->N*sizeof(double));
	AA_d = malloc(gb->NP*sizeof(double *));
	for (i=0; i<gb->NP; i++) AA_d[i] = malloc(2*gb->N*sizeof(double));
	EE_d = malloc(gb->NP*sizeof(double *));
	for (i=0; i<gb->NP; i++) EE_d[i] = malloc(2*gb->N*sizeof(double));
	
	for (i=0; i<2*N; i++)
	{
		XX_p[i] = 0.;
		AA_p[i] = 0.;
		EE_p[i] = 0.;
		XX_m[i] = 0.;
		AA_m[i] = 0.;
		EE_m[i] = 0.;
	}
	
	for (i=0; i<NP; i++)
	{
		gb_p->params[i] = 0.;
		gb_m->params[i] = 0.;
	}
	
	for (i=0; i<gb->NP; i++) 
	{
		for(j=0; j<2*gb->N; j++) 
		{
			XX_d[i][j] = 0.;
			AA_d[i][j] = 0.;
			EE_d[i][j] = 0.;
		}
	}
	
	// calculate the derivatives as a function of frequency
	for (i=0; i<NP; i++)
	{
		for (j=0; j<NP; j++)
		{
			gb_p->params[j] = gb->params[j];
			gb_m->params[j] = gb->params[j];
		}

		gb_p->params[i] += ep;
		gb_m->params[i] -= ep;
		
		FAST_LISA_GR(gb_p->params, gb_p->N, XX_p, AA_p, EE_p, gb_p->NP);
		FAST_LISA_GR(gb_m->params, gb_m->N, XX_m, AA_m, EE_m, gb_m->NP);
		
		for (j=0; j<2*N; j++) 
		{
			XX_d[i][j] = (XX_p[j] - XX_m[j])/(2.0*ep);
			AA_d[i][j] = (AA_p[j] - AA_m[j])/(2.0*ep);
			EE_d[i][j] = (EE_p[j] - EE_m[j])/(2.0*ep);
		}
	}
	free(XX_p); free(AA_p); free(EE_p);
	free(XX_m); free(AA_m); free(EE_m);
	free(gb_p->params); free(gb_m->params);
	free(gb_p); free(gb_m);
	
	
	// NWIP the derivative arrays
	for (i=0; i<NP; i++)
	{
		for(j=i; j<NP; j++)
		{
			arg = 0.;
			for (m=0; m<N; m++)
			{
				k = (gb->q + m - gb->N/2);    
				f = (double)(k)/Tobs; 			
				instrument_noise(f, &SnAE, &SnX);
		
				iRe = 2*m;		
				iIm = 2*m+1;   
		
				arg +=  (AA_d[i][iRe]*AA_d[j][iRe] + AA_d[i][iIm]*AA_d[j][iIm] 
					    +EE_d[i][iRe]*EE_d[j][iRe] + EE_d[i][iIm]*EE_d[j][iIm])/SnAE;
			}
			arg *= 4.*Tobs;
			
			Fisher[i][j] = arg;
		}
	}
	for(i=0; i<NP; i++)
	{
		free(XX_d[i]);
		free(AA_d[i]);
		free(EE_d[i]);
	} 
	free(XX_d); free(AA_d); free(EE_d);
	
	// make use of symmetry
	for (i=0; i<NP; i++)
	{
		for (j=i+1; j<NP; j++)
		{
			Fisher[j][i] = Fisher[i][j];
		}
	}
	return;
}

void matrix_eigenstuff(double **matrix, double **evector, double *evalue, int N)
{
	int i,j;

	// Don't let errors kill the program (yikes)
	gsl_set_error_handler_off();
	int err=0;

	// Find eigenvectors and eigenvalues
	gsl_matrix *GSLfisher = gsl_matrix_alloc(N,N);
	gsl_matrix *GSLcovari = gsl_matrix_alloc(N,N);
	gsl_matrix *GSLevectr = gsl_matrix_alloc(N,N);
	gsl_vector *GSLevalue = gsl_vector_alloc(N);

	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++)
		{
			if(matrix[i][j]!= matrix[i][j])fprintf(stderr,"WARNING: nan matrix element, now what?\n");
			gsl_matrix_set(GSLfisher, i, j, matrix[i][j]);
		}
	}

	// sort and put them into evec
	gsl_eigen_symmv_workspace * workspace = gsl_eigen_symmv_alloc (N);
	gsl_permutation * permutation = gsl_permutation_alloc(N);
	err += gsl_eigen_symmv (GSLfisher, GSLevalue, GSLevectr, workspace);
	err += gsl_eigen_symmv_sort (GSLevalue, GSLevectr, GSL_EIGEN_SORT_ABS_ASC);

	// eigenvalues destroy matrix
	for(i=0; i<N; i++) for(j=0; j<N; j++) gsl_matrix_set(GSLfisher, i, j, matrix[i][j]);

	err += gsl_linalg_LU_decomp(GSLfisher, permutation, &i);
	err += gsl_linalg_LU_invert(GSLfisher, permutation, GSLcovari);

	if(err>0)
	{
		for(i=0; i<N; i++)for(j=0; j<N; j++)
		{
			evector[i][j] = 0.0;
			if(i==j)
			{
				evector[i][j] = 1.0;
				evalue[i]     = 1./matrix[i][j];
			}
		}

	}
	else
	{	
		for(i=0; i<N; i++)
		{	//unpack arrays from gsl inversion
			evalue[i] = gsl_vector_get(GSLevalue, i);
			for(j=0; j<N; j++)
			{
				evector[i][j] = gsl_matrix_get(GSLevectr, i, j);
				if(evector[i][j] != evector[i][j]) evector[i][j] = 0.;
			}
		}
			
		for(i=0; i<N; i++)
		{	//copy covariance matrix back into Fisher
			for(j=0; j<N; j++)
			{
				matrix[i][j] = gsl_matrix_get(GSLcovari, i, j);
			}
		}

		for(i=0; i<N; i++)
		{	//cap minimum size eigenvalues
			if(evalue[i] != evalue[i] || evalue[i] <= 10.0) evalue[i] = 10.;
			//fprintf(stdout, "here\n");
		}
	}

	gsl_vector_free(GSLevalue);
	gsl_matrix_free(GSLfisher);
	gsl_matrix_free(GSLcovari);
	gsl_matrix_free(GSLevectr);
	gsl_eigen_symmv_free(workspace);
	gsl_permutation_free(permutation);
}

double invert_matrix(double **matrix, int N)
{
	int i,j;
	double cond;

	// Don't let errors kill the program (yikes)
	gsl_set_error_handler_off ();
	int err=0;

	// Find eigenvectors and eigenvalues
	gsl_matrix *GSLmatrix = gsl_matrix_alloc(N,N);
	gsl_matrix *GSLinvrse = gsl_matrix_alloc(N,N);
	gsl_matrix *cpy		  = gsl_matrix_alloc(N,N);
	gsl_matrix *SVDinv	  = gsl_matrix_alloc(N,N);
	gsl_matrix *Dmat	  = gsl_matrix_alloc(N,N);
	gsl_matrix *temp      = gsl_matrix_alloc(N, N);

	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++)
		{
			if(matrix[i][j]!=matrix[i][j])
			{
				fprintf(stdout, "error for parameters (%d, %d)\n", i, j);
				fprintf(stderr,"GalacticBinaryMath.c:172: WARNING: nan matrix element, now what?\n");
			}
			gsl_matrix_set(GSLmatrix,i,j,matrix[i][j]);
			gsl_matrix_set(cpy,i,j,matrix[i][j]);
		}
	}

	//////
	//
	//	Calculate the SVD and condition number
	//
	///////

	gsl_matrix *V = gsl_matrix_alloc (N,N);
	gsl_vector *D = gsl_vector_alloc (N);
	gsl_vector *work = gsl_vector_alloc (N);

	gsl_linalg_SV_decomp(cpy, V, D, work);


	double max, min;
	max = -0.1;
	min = INFINITY;

	for (i=0; i<N; i++)
	{
		if (gsl_vector_get(D,i) > max) max = gsl_vector_get(D,i);

		if (gsl_vector_get(D,i) < min) min = gsl_vector_get(D,i);
	}

	cond = log10(max/min);
	
	
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++) 
		{
			if (i == j)
			{
				if (gsl_vector_get(D,i) < 1.0e-10) 
				{
					fprintf(stdout, "Near Singular value!!! ---> %e\n", gsl_vector_get(D,i));
					gsl_matrix_set(Dmat, i, j, 0.);
				} else
				{
					gsl_matrix_set(Dmat, i, j, 1./gsl_vector_get(D,i));
				}
				
			} else 
			{
				gsl_matrix_set(Dmat, i, j, 0.);
			}
		}
	
	}

	gsl_matrix_transpose(cpy);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Dmat, cpy,   0.0, temp);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V, temp, 0.0, SVDinv);
	
	
	////////

	gsl_permutation * permutation = gsl_permutation_alloc(N);

	err += gsl_linalg_LU_decomp(GSLmatrix, permutation, &i);
	err += gsl_linalg_LU_invert(GSLmatrix, permutation, GSLinvrse);

	if(err>0)
	{
		fprintf(stderr,"GalacticBinaryMath.c:184: WARNING: singluar matrix\n");
		fflush(stderr);
	}else
	{
		//copy covariance matrix back into Fisher
		for(i=0; i<N; i++)
		{
			for(j=0; j<N; j++) 
			{
				//matrix[i][j] = gsl_matrix_get(GSLinvrse,i,j);
				matrix[i][j] = gsl_matrix_get(SVDinv, i, j);
			}
		}
	}

	gsl_vector_free(D);
	gsl_vector_free(work);
	gsl_matrix_free(V);
	gsl_matrix_free(Dmat);
	gsl_matrix_free(SVDinv);
	gsl_matrix_free(temp);
	gsl_matrix_free(cpy);

	gsl_matrix_free (GSLmatrix);
	gsl_matrix_free (GSLinvrse);
	gsl_permutation_free (permutation);

	return cond;
}

void prior_jump(struct GB *gb_y, gsl_rng *r, double f0)
{
	double f_lo, f_up;
	const int NP = gb_y->NP;
	
	// HACK
	f_lo = f0 - gb_y->N/Tobs;
	f_up = f0 + gb_y->N/Tobs;
	
	gb_y->params[0] = f_lo*Tobs + gsl_rng_uniform(r)*(f_up*Tobs - f_lo*Tobs);
	gb_y->params[1] = -1. + gsl_rng_uniform(r)*2.;
	gb_y->params[2] = gsl_rng_uniform(r)*PI2;
	gb_y->params[3] = -60. + gsl_rng_uniform(r)*30.;
	gb_y->params[4] = -1. + gsl_rng_uniform(r)*2.;
	gb_y->params[5] = gsl_rng_uniform(r)*M_PI;
	gb_y->params[6] = gsl_rng_uniform(r)*PI2;
	if (NP > 7) gb_y->params[7] = -30. + gsl_rng_uniform(r)*60.;
	if (NP > 8) gb_y->params[8] = -30. + gsl_rng_uniform(r)*60.;
 
 
 	gb_y->q = (long)gb_y->params[0];
 	
	return;
}

void diff_ev_jump(struct GB *gb_x, struct GB *gb_y, gsl_rng *r, double ***history, long im, int ic)
{
	int i, j, k;
		
	double alpha, beta;
	const int NP = gb_x->NP;
	
	if (im<2) 
	{
		for (i=0; i<NP; i++) gb_y->params[i] = gb_x->params[i]; // i.e. change nothing
	} else 
	{	
		// select two different points
		j = (int)( (double)im*gsl_rng_uniform(r) );
		do {
			k = (int)( (double)im*gsl_rng_uniform(r) );
		} while (j==k);
		
		alpha = 1.0;
		beta = gsl_rng_uniform(r);
		if (beta < 0.9) alpha = gsl_ran_gaussian(r, 1.);
		
		for (i=0; i<NP; i++)
		{
			gb_y->params[i] = gb_x->params[i] + alpha*(history[ic][j][i] - history[ic][k][i]);
		}
	}
	
	gb_y->q = (long)(gb_y->params[0]);
	
	return;
}

void Fisher_jump(struct GB *gb_y, struct GB *gb_x, gsl_rng *r, double **evecs, double *evals, double heat)
{
	int j, k;
	const int NP = gb_x->NP;
	
	double *jump;
	jump = malloc(NP*sizeof(double));
	
	k = (int)(gsl_rng_uniform(r)*(double)NP);
	for (j=0; j<NP; j++)
	{
		jump[j] = gsl_ran_gaussian(r, 1.)*evecs[j][k]/sqrt(evals[k]*NP/heat); 
	}
	
	//check jump value, set to small value if singular
  	for(j=0; j<NP; j++) if(jump[j]!=jump[j]) jump[j] = 0.01*gb_x->params[j];
  	
	for (j=0; j<NP; j++)
	{
		gb_y->params[j] = gb_x->params[j] + jump[j]; 

	}
	gb_y->q = (long)(gb_y->params[0]);


	for(int j=0; j<NP; j++)
	{
		if(gb_y->params[j] != gb_y->params[j]) fprintf(stderr,"Fisher draw error\n");
	}
	
	free(jump);

	return;
}

void check_priors(double *params, int *meets_priors, int NP, double f0, long N)
{
	double f_lo, f_up;
	
	// HACK
	f_lo = f0 - N/Tobs;
	f_up = f0 + N/Tobs;
 	
	if (params[0] < f_lo*Tobs  || params[0] > f_up*Tobs) 
	{
		*meets_priors = 0;
		return;
	}
	if (params[1] < -1.        || params[1] > 1.)
	{
		*meets_priors = 0;
		return;
	}
	if (params[2] < 0.         || params[2] > PI2) 
	{
		*meets_priors = 0;
		return;
	}      
	if (params[3] < -60. 	   || params[3] > -30.) 
	{
		*meets_priors = 0;
		return;
	}	
	if (params[4] < -1.	       || params[4] > 1.) 	
	{
		*meets_priors = 0;
		return;
	}	 
	if (params[5] < 0. 	       || params[5] > M_PI) 	
	{
		*meets_priors = 0;
		return;
	} 
	if (params[6] < 0. 	       || params[6] > PI2) 		
	{
		*meets_priors = 0;
		return;
	}
	
	if (NP > 7)
	{
		if (params[7] < -1000.   || params[7] > 1000.) 		
		{
			*meets_priors = 0;
			return;
		}     
	}
	if (NP > 8) 
	{
		if (params[8] < -1000.   || params[8] > 1000.) 
		{
			*meets_priors = 0;
			return;
		}    		   
	}
	
	*meets_priors = 1;
	
	return;
}

#undef PBSTR 
#undef PBWIDT


