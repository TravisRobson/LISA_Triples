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

void printProgress(double percentage, double acceptance, double logL)
{
    double val = (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3.1f%% [%.*s%*s] acceptance rate: %f, logL: %f", val, lpad, PBSTR, rpad, "", acceptance, logL);
    fflush (stdout);
}

double get_mb(double Mchirp, double ma);

double get_f_src(struct GB *gb, double t);
double get_f0H(struct GB *gb, struct Triple *trip, double t);
void get_f0H_derivs(struct GB *gb, struct Triple *trip, double *dfdt_0H, double *d2fdt2_0H);

double get_logL(struct GB *gb, double *AA, double *EE, double *AALS, double *EELS);

void matrix_eigenstuff(double **matrix, double **evector, double *evalue, int N);
//void invert_matrix(double **matrix, int N);
double invert_matrix(double **matrix, int N);
void check_priors(double *params, int *meets_priors, int NP);
void calc_TayExp_Fisher(struct GB *gb, double **Fisher);
void setup_triple(struct Triple *trip, char *argv[]);
double get_snrAE(struct GB *gb, double *AA, double *EE);
double get_overlap(struct GB *gb, double *AA, double *EE, double *AALS, double *EELS);
void print_signal(struct GB *gb, FILE *fptr, double *XX, double *AA, double *EE);
void fill_data_stream(struct GB *gb, double *XX, double *AA, double *EE, double *XXLS, double *AALS, double *EELS, long NFFT);
void copy_gb(struct GB *dest, struct GB *src);
void Fisher_jump(struct GB *gb_y, struct GB *gb_x, gsl_rng *r, double **evecs, double *evals, double heat);
void diff_ev_jump(struct GB *gb_x, struct GB *gb_y, gsl_rng *r, double ***history, long m, int c);
void prior_jump(struct GB *gb_y, gsl_rng *r);

int main(int argc, char *argv[])
{
	int p; 
	const int NC = 8;
	int c = 0;
	
	long i, j, m, k;
	const long NFFT = (long)(Tobs/dt);
	long NMCMC;
	long seed;
	
	double f0, dfdt_0, co_lat, phi, amp, iota, psi, phi0;
	double snr, snr_max;
	double logL, logLy, loga, logL_max;
	double FF, time_spent;
	double alpha;
	
	double *logLx;
	double   *XXLS, *AALS, *EELS;
	double   *XX,   *AA,   *EE;
	double  **Fisher_AE;
	double ***history;
	
	FILE *in_file, *out_file;
	
	fprintf(stdout, "==============================================================\n\n");
	
	clock_t begin = clock();
	
	fprintf(stdout, "Observation T: %.3f yrs\n", Tobs/YEAR);

	// read in test source parameters
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
	
	p = 7;
	struct GB *gb = malloc(sizeof(struct GB));
	gb->N  = pow(2, p);
	gb->T  = Tobs;
	gb->NP = 9;    // f0*T, cos(theta), phi, amp, cos(iota), psi, phi0, dfdt_0*T*T, d2fdt2_0*T*T*T
	gb->q  = (long)(f0*gb->T);
	gb->params = malloc(gb->NP*sizeof(double));
	
	// set the parameters
	gb->params[0] = f0*Tobs;
	gb->params[1] = cos(PIon2 - co_lat); // convert to spherical polar
	gb->params[2] = phi;
	gb->params[3] = log(amp);
	gb->params[4] = cos(iota);
	gb->params[5] = psi;
	gb->params[6] = phi0;
	if (gb->NP > 7) gb->params[7] = dfdt_0*Tobs*Tobs;
	if (gb->NP > 8) gb->params[8] = 11./3.*dfdt_0*dfdt_0/f0*Tobs*Tobs*Tobs;
	
	struct Triple *trip = malloc(sizeof(struct Triple));
	setup_triple(trip, argv);
	
	//////// Need to construct a GB and triple shit for the data
	struct GB *gb_data = malloc(sizeof(struct GB));
	gb_data->N  = pow(2,p);
	gb_data->T  = Tobs;
	gb_data->NP = 8;    
	gb_data->q  = (long)(f0*gb->T); //(long)(f0H*gb->T);
	gb_data->params = malloc(gb_data->NP*sizeof(double));
	
	// set the parameters
	gb_data->params[0] = f0*Tobs;
	gb_data->params[1] = cos(PIon2 - co_lat); // convert to spherical polar
	gb_data->params[2] = phi;
	gb_data->params[3] = log(amp);
	gb_data->params[4] = cos(iota);
	gb_data->params[5] = psi;
	gb_data->params[6] = phi0;
	gb_data->params[7] = dfdt_0*Tobs*Tobs;
	//////////////////
	
	XX = malloc(2*gb->N*sizeof(double));
	AA = malloc(2*gb->N*sizeof(double));
	EE = malloc(2*gb->N*sizeof(double));
	
	GB_triple(gb_data, XX, AA, EE, trip);

	out_file = fopen("test_signal.dat" ,"w");
	print_signal(gb_data, out_file, XX, AA, EE);
	
	fprintf(stdout, "NFFT: %ld, 2^{p}, p = %ld\n", NFFT, (long)(log(NFFT)/log(2.)));
	
	XXLS = malloc(NFFT*sizeof(double));
	AALS = malloc(NFFT*sizeof(double));
	EELS = malloc(NFFT*sizeof(double));
	
	snr = get_snrAE(gb_data, AA, EE); // fprintf(stdout, "\nAE SNR: %f\n", snr);
	// adjust to get an SNR of 15
	for (i=0; i<gb->N; i++)
	{
		AA[2*i]   *= 15./snr;
		AA[2*i+1] *= 15./snr;
		EE[2*i]   *= 15./snr;
		EE[2*i+1] *= 15./snr;
	}
	snr = get_snrAE(gb_data, AA, EE); fprintf(stdout, "\nAE SNR: %f\n", snr);
	fill_data_stream(gb_data, XX, AA, EE, XXLS, AALS, EELS, NFFT);
	
	FAST_LISA(gb->params, gb->N, XX, AA, EE, gb->NP);
	gb->params[3] = log(amp*15./get_snrAE(gb, AA, EE));
	// now calculate the first best guess
 	
        

// 		gb->params[0]=9.98761323e+04 ;
// 	gb->params[1]=4.90312853e-01;
//    gb->params[2]= 5.26463018e+00  ;
//    gb->params[3]= -5.19308985e+01  ;
//    gb->params[4]= 1.93483086e-01;
//    gb->params[5] =  2.52755578e+00;
//    gb->params[6]= 4.69430206e+00 ;
//     gb->params[7]= 5.71504209e+00 ;  
//     gb->params[8]= 3.95360309e-01;
//     gb->q = (long)gb->params[0];
    FAST_LISA(gb->params, gb->N, XX, AA, EE, gb->NP);
    double first_snr = get_snrAE(gb, AA, EE);
//     	out_file = fopen("max_signal.dat" ,"w");
// 	print_signal(gb, out_file, XX, AA, EE);
     
 	logL = get_logL(gb, AA, EE, AALS, EELS); fprintf(stdout, "\nAE logL: %f, first snr: %f\n", logL, first_snr);
 	
	FF = get_overlap(gb, AA, EE, AALS, EELS)/snr/get_snrAE(gb, AA, EE); fprintf(stdout, "\nFF: %f\n", FF);
	
	logLx = malloc(NC*sizeof(double));
	for (c=0; c<NC; c++) logLx[c] = logL;

	Fisher_AE = malloc(gb->NP*sizeof(double *));
	for (i=0; i<gb->NP; i++) Fisher_AE[i] = malloc(gb->NP*sizeof(double));
	
	for (i=0;i<gb->NP;i++)
	{
		for (j=0;j<gb->NP;j++) Fisher_AE[i][j] = 0.;
	}
	
	calc_TayExp_Fisher(gb, Fisher_AE);
	
	double **AE_Fish_evec;
	double  *AE_Fish_eval;
	
	AE_Fish_evec = malloc(gb->NP*sizeof(double *));
	for (i=0; i<gb->NP; i++) AE_Fish_evec[i] = malloc(gb->NP*sizeof(double));
	AE_Fish_eval = malloc(gb->NP*sizeof(double));
	for (i=0; i<gb->NP; i++) 
	{
		for (j=0; j<gb->NP; j++)  AE_Fish_evec[i][j] = 0.;
		AE_Fish_eval[i] = 0.;
	}

	double cond;
	cond = invert_matrix(Fisher_AE, gb->NP); fprintf(stdout, "Condition number: %e\n\n", cond);
	
	matrix_eigenstuff(Fisher_AE, AE_Fish_evec, AE_Fish_eval, gb->NP);
	
	NMCMC = 1e5;
	seed  = atoi(argv[1]); 
	
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);
	gsl_rng_set(r, seed);
	
	gsl_rng **r_arr = malloc(NC*sizeof(gsl_rng *));
	for (c=0; c<NC; c++) r_arr[c] = gsl_rng_alloc(T);
	for (c=0; c<NC; c++) gsl_rng_set(r_arr[c], c*seed);
	
	
	struct GB **gb_x_arr = malloc(NC*sizeof(struct GB *));
	for (c=0; c<NC; c++) gb_x_arr[c] = malloc(sizeof(struct GB));
	for (c=0; c<NC; c++)
	{
		gb_x_arr[c]->params = malloc(gb->NP*sizeof(double));
		copy_gb(gb_x_arr[c], gb);
	}
		
	struct GB *gb_y = malloc(sizeof(struct GB));
	gb_y->params = malloc(gb->NP*sizeof(double));
	copy_gb(gb_y, gb);
	
	out_file = fopen(argv[2], "w");
	int meets_priors = 1;
	
	//double heat = 0.04*snr*snr;
	double *heat;
	heat = malloc(NC*sizeof(double));
	
	int *who;
	who = malloc(NC*sizeof(int));
	for (c=0; c<NC; c++) 
	{
		who[c] = c;
		//gb_x_arr[c]->who = c;
	}
	    
    struct GB *gb_max = malloc(sizeof(struct GB));
    gb_max->params = malloc(gb->NP*sizeof(double));
    copy_gb(gb_max, gb);

	logLy    = -1.0e30;
	logL_max = -1.0e30;
	
	history = malloc(NC*sizeof(double **));
	for (i=0; i<NC; i++) history[i] = malloc(NMCMC/10*sizeof(double *));
	for (i=0; i<NC; i++)
	{
		for (j=0; j<NMCMC/10; j++) history [i][j] = malloc(gb->NP*sizeof(double));
	}	
	
	int mcount  = 0;
	int scount  = 0;
	int maccept = 0;
	int saccept = 0;
	double beta, jump; int hold, id;
	
	double *cnt, *acc;
	cnt = malloc(NC*sizeof(double));
	acc = malloc(NC*sizeof(double));
	for (c=0; c<NC-1; c++)
	{	
		cnt[c] = 0;
		acc[c] = 0;
	}
	
	double nu = 100.;
	double t0 = 1.e4;
	double *S = malloc(NC*sizeof(double));
	double **A = malloc(NC*sizeof(double *));
	for (c=0; c<NC; c++) A[c] = malloc(2*sizeof(double));
			
	for (m=0; m<NMCMC; m++)
	{
		//heat[0] = pow(10.0, -3.*(double)(m)/(double)(NMCMC));
		heat[0] = 4.*pow(10.0, -3.6*(double)(m)/(double)(NMCMC));
		for (c=1; c<NC; c++) heat[c] = heat[c-1]*1.2; //if (m==0) 
// 		for (c=0; c<NC; c++) heat[c] = pow(10.0, -3.0*(double)(m)/(double)(NMCMC)*(double)(NC-1-c)/(double)(NC-1));
		
		alpha = gsl_rng_uniform(r);
		
		if ((m+1)%(int)(0.01*NMCMC) == 0.) printProgress((double)(m+1)/(double)NMCMC, (double)maccept/(double)mcount, logLx[who[0]]);
		if (m == NMCMC-1) printProgress((double)(m+1)/(double)NMCMC, (double)maccept/(double)mcount, logLx[who[0]]);	
		
		// propose chain swap
		if (NC > 1 && alpha < 0.5)
		{	
			scount += 1;
			alpha = (double)(NC-1)*gsl_rng_uniform(r);
			k = (int)(alpha);
			if (m>0e4) cnt[k]++;
			beta  = (logLx[who[k]] - logLx[who[k+1]])/heat[k+1];
			beta -= (logLx[who[k]] - logLx[who[k+1]])/heat[k];
		
			alpha = log(gsl_rng_uniform(r));

			if (beta > -INFINITY && beta > alpha)
			{
				hold     = who[k];
				who[k]   = who[k+1];
				who[k+1] = hold;
				saccept++;
				if (m>0e4) acc[k]++;
			}
		}
		
		else // do a MCMC update
		{	
			mcount += 1;
			for (c=0; c<NC; c++)
			{
				id = who[c];
			
				jump = gsl_rng_uniform(r);
				if (jump < 0.01) prior_jump(gb_y, r);
				else if (jump < 0.5 && jump > 0.01) Fisher_jump(gb_y, gb_x_arr[id], r_arr[c], AE_Fish_evec, AE_Fish_eval, heat[c]);
				else diff_ev_jump(gb_x_arr[id], gb_y, r_arr[c], history, m, c);
				//Fisher_jump(gb_y, gb_x_arr[id], r_arr[c], AE_Fish_evec, AE_Fish_eval, heat[c]);
				
				check_priors(gb_y->params, &meets_priors, gb_y->NP);
				if (meets_priors != meets_priors) fprintf(stderr, "[%ld] wtf meets_priors\n", m);
				
				for (j=0; j<gb_y->NP; j++) if (gb_y->params[j] != gb_y->params[j]) fprintf(stderr, "[%ld] WTF\n", m);
				
				if (meets_priors == 1)
				{
					FAST_LISA(gb_y->params, gb_y->N, XX, AA, EE, gb_y->NP); 
					logLy = get_logL(gb_y, AA, EE, AALS, EELS);		
					loga = log(gsl_rng_uniform(r));
					
					if (logLy != logLy) fprintf(stderr, "[%ld] WTF logLy...\n", m);
					
					if (logLy > -INFINITY && loga < (logLy - logLx[id])/heat[c])
					{
						if (c==0) maccept += 1.;   
						gb_x_arr[id]->q = gb_y->q;
						for (j=0; j<gb->NP; j++) gb_x_arr[id]->params[j] = gb_y->params[j];
						logLx[id] = logLy;
					
						// store the max likelihood values
						if (c==0 && logLx[who[0]] > logL_max) // only do for cold chain
						{
							logL_max   = logLx[who[0]];
							gb_max->q  = gb_x_arr[who[0]]->q;
							for (j=0; j<gb->NP; j++) gb_max->params[j] = gb_x_arr[who[0]]->params[j];
						}
					}
				}
			}
		}
		
		if (m%10 == 0)
		{	
			for (c=0; c<NC; c++)
			{
				for (j=0; j<gb->NP; j++) history[c][m/10][j] = gb_x_arr[who[c]]->params[j]; 
			}
		}
// 		if (m%100 == 0) // adjust temperature ladder
// 		{
// 			for (c=1; c<NC-1; c++)
// 			{
// 				S[c]    = log(heat[c] - heat[c-1]);
// 				A[c][0] = (double)acc[c-1]/(double)cnt[c-1];
// 				A[c][1] = (double)acc[c]/(double)cnt[c];
// 			}	
// 			for (c=1; c<NC; c++)
// 			{
// 				S[c] = (A[c][0] - A[c][1])*(t0/((double)NMCMC + t0))/nu;
// 				heat[c] = heat[c-1] + exp(S[c]);
// 				if (heat[c]/heat[c-1] < 1.01) heat[c] = 1.01*heat[c-1];
// 			}	
// 		}
		
		if (m%100 == 0)
		{
			calc_TayExp_Fisher(gb_max, Fisher_AE);
			cond = invert_matrix(Fisher_AE, gb_max->NP);
			matrix_eigenstuff(Fisher_AE, AE_Fish_evec, AE_Fish_eval, gb_max->NP);
		}
	}
	
	fprintf(stdout, "\nswap accept: %f\n", (double)saccept/(double)scount);
	fprintf(stdout, "mcmc accept: %f\n\n", (double)maccept/(double)mcount);
	saccept = 0; maccept = 0; mcount = 0; scount = 0;
	
	for (c=0; c<NC-1; c++)
	{
		fprintf(stdout, "swap[%d] rate: %f\n", c, (double)acc[c]/(double)cnt[c]);
	}
	fprintf(stdout, "\n");
	for (c=0; c<NC-1; c++)
	{
		fprintf(stdout, "logLx[%d]: %f\n", c, logLx[who[c]]);
	}
	for (c=0; c<NC-1; c++)
	{	
		cnt[c] = 0;
		acc[c] = 0;
	}

	meets_priors = 1; // reset	
	
	if (m%1000 == 0)
	{
		calc_TayExp_Fisher(gb_max, Fisher_AE);
		cond = invert_matrix(Fisher_AE, gb_max->NP);
		matrix_eigenstuff(Fisher_AE, AE_Fish_evec, AE_Fish_eval, gb_max->NP);
		fprintf(stdout, "\nheat[0]: %f, logL_max: %f\n", heat[0], logL_max);
	}
	
	for (i=0; i<NC; i++) for (j=0; j<NMCMC/10; j++) free(history[i][j]);
	
	calc_TayExp_Fisher(gb_max, Fisher_AE);
	cond = invert_matrix(Fisher_AE, gb->NP); fprintf(stdout, "\n\nCondition number: %e\n\n", cond);
	matrix_eigenstuff(Fisher_AE, AE_Fish_evec, AE_Fish_eval, gb->NP);
	
	NMCMC = 1e5;
	//NC = 7;
	
	history = malloc(NC*sizeof(double **));
	for (i=0; i<NC; i++) history[i] = malloc(NMCMC/10*sizeof(double *));
	for (i=0; i<NC; i++)
	{
		for (j=0; j<NMCMC/10; j++) history [i][j] = malloc(gb->NP*sizeof(double));
	}	
		
	heat[0] = 1.;
	for (c=1; c<NC; c++) heat[c] = heat[c-1]*1.8;
	mcount = 0; scount = 0; maccept = 0; saccept = 0;
	
// 	FILE *log_file;
// 	log_file = fopen("logL.dat", "w");
	
	for (c=0; c<NC; c++) 
	for (c=0; c<NC; c++)
	{
		S[c] = 0.;
		A[c][0] = 0.;
		A[c][1] = 0.;
	}
	
	nu = 1.;
	t0 = 1.e4;

	
	for (m=0; m<NMCMC; m++)
	{
		if ((m+1)%(int)(0.01*NMCMC) == 0.) printProgress((double)(m+1)/(double)NMCMC, (double)maccept/(double)mcount, logLx[who[0]]);
		if (m == NMCMC-1)printProgress((double)(m+1)/(double)NMCMC, (double)maccept/(double)mcount, logLx[who[0]]);	
		alpha = gsl_rng_uniform(r);
		
		// propose chain swap
		if (NC > 1 && alpha < 0.5)
		{	
			scount++;
			alpha = (double)(NC-1)*gsl_rng_uniform(r);
			k = (int)(alpha);
			if (m>0e4) cnt[k]++;
			beta  = (logLx[who[k]] - logLx[who[k+1]])/heat[k+1];
			beta -= (logLx[who[k]] - logLx[who[k+1]])/heat[k];
		
			alpha = log(gsl_rng_uniform(r));

			if (beta > -INFINITY && beta > alpha)
			{
				hold     = who[k];
				who[k]   = who[k+1];
				who[k+1] = hold;
				saccept++;
				if (m>0e4) acc[k]++;
			}
		}
		
		else // do a MCMC update
		{
			mcount++;
			for (c=0; c<NC; c++)
			{
				id = who[c];

				jump = gsl_rng_uniform(r);

				if (jump < 0.0) prior_jump(gb_y, r);
				else if (jump < 0.5 && jump > 0.00) Fisher_jump(gb_y, gb_x_arr[id], r_arr[c], AE_Fish_evec, AE_Fish_eval, heat[c]);
				else diff_ev_jump(gb_x_arr[id], gb_y, r_arr[c], history, m, c);
				//Fisher_jump(gb_y, gb_x_arr[id], r_arr[c], AE_Fish_evec, AE_Fish_eval, heat[c]);
				
				for (j=0; j<gb_y->NP; j++) if (gb_y->params[j] != gb_y->params[j]) fprintf(stderr, "[%ld] WTF\n", m);
				
				check_priors(gb_y->params, &meets_priors, gb_y->NP);
				if (meets_priors != meets_priors) fprintf(stderr, "[%ld] wtf meets_priors\n", m);
				logLy = -1.0e30;
				if (meets_priors == 1)
				{
					FAST_LISA(gb_y->params, gb_y->N, XX, AA, EE, gb_y->NP); 
					logLy = get_logL(gb_y, AA, EE, AALS, EELS);		
					if (logLy != logLy) fprintf(stderr, "[%ld] WTF logLy...\n", m);
					
					loga = log(gsl_rng_uniform(r));

					if (logLy > -INFINITY && loga < (logLy - logLx[id])/heat[c])
					{
						if (c==0) maccept++;   
						gb_x_arr[id]->q  = gb_y->q;
						for (j=0; j<gb->NP; j++) gb_x_arr[id]->params[j] = gb_y->params[j];
						logLx[id] = logLy;
					
						// store the max likelihood values
						if (logLx[who[0]] > logL_max && c == 0)
						{   // only for cold chain
							logL_max   = logLx[who[0]];
							gb_max->q  = gb_x_arr[who[0]]->q;
							for (j=0; j<gb->NP; j++) gb_max->params[j] = gb_x_arr[who[0]]->params[j];
						}
					}
				}
			}
		}
		if (m%10 == 0)
		{	for (c=0; c<NC; c++)
			{
				for (j=0; j<gb->NP; j++) history[c][m/10][j] = gb_x_arr[who[c]]->params[j]; 
			}
			
			// print cold chain info
			fprintf(out_file, "%ld %.12g ", m/10, logLx[who[0]]);
			for (j=0; j<gb->NP; j++) fprintf(out_file, "%.12g ", gb_x_arr[who[0]]->params[j]);
			fprintf(out_file, "\n");
			
			// print different log-likelihoods
// 			fprintf(log_file, "%ld %.12g ", m/10, logLx[who[0]]);
// 			for (c=1; c<NC; c++) fprintf(log_file, "%.12g ", logLx[who[c]]);
// 			fprintf(log_file, "\n");
		}
		
		if (m%100 == 0) // adjust temperature ladder
		{
			for (c=1; c<NC-1; c++)
			{
				S[c]    = log(heat[c] - heat[c-1]);
				A[c][0] = (double)acc[c-1]/(double)cnt[c-1];
				A[c][1] = (double)acc[c]/(double)cnt[c];
			}	
			for (c=1; c<NC; c++)
			{
				S[c] = (A[c][0] - A[c][1])*(t0/((double)NMCMC + t0))/nu;
				heat[c] = heat[c-1] + exp(S[c]);
				if (heat[c]/heat[c-1] < 1.01) heat[c] = 1.01*heat[c-1];
			}	
		}
		
		if (m%100 == 0)
		{
			calc_TayExp_Fisher(gb_max, Fisher_AE);
			cond = invert_matrix(Fisher_AE, gb_max->NP);
			matrix_eigenstuff(Fisher_AE, AE_Fish_evec, AE_Fish_eval, gb_max->NP);
		}
	}
	
	fclose(out_file);
	// fclose(log_file);
	for (i=0; i<NC; i++) for (j=0; j<NMCMC/10; j++) free(history[i][j]);

	fprintf(stdout, "\n\nswap accept: %.2f%%\n", 100.*(double)saccept/(double)scount);
	fprintf(stdout, "mcmc accept: %.2f%%\n", 100.*(double)maccept/(double)mcount);
	fprintf(stdout, "logL_max: %f\n", logL_max);
	
	for (c=0; c<NC-1; c++)
	{
		fprintf(stdout, "swap[%d] rate: %f\n", c, (double)acc[c]/(double)cnt[c]);
	}
	fprintf(stdout, "\n");
	for (c=0; c<NC; c++)
	{
		fprintf(stdout, "heat[%d]: %f\n", c, heat[c]);
	}
	
	// calculate max-likelihood signal
	FAST_LISA(gb_max->params, gb_max->N, XX, AA, EE, gb_max->NP);
	snr_max = get_snrAE(gb_max, AA, EE); fprintf(stdout, "\nAE max SNR: %f\n", snr_max);														
							
	FF = get_overlap(gb_max, AA, EE, AALS, EELS)/snr/snr_max;  fprintf(stdout, "\nFitting Factor: %f\n", FF);
	
	clock_t end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	fprintf(stdout, "\nSimulated Annealing runtime: %f sec\n", time_spent);
		
		
	FAST_LISA(gb_max->params, gb_max->N, XX, AA, EE, gb_max->NP);
	out_file = fopen("max_signal.dat" ,"w");
	print_signal(gb_max, out_file, XX, AA, EE);
	
	fprintf(stdout, "\nf0xTobs dif:           %e\n", f0*Tobs - gb_max->params[0]);
	fprintf(stdout, "dfdt_0xTobs^{2} dif:   %e\n",   dfdt_0*Tobs*Tobs - gb_max->params[7]);
	fprintf(stdout, "df2dt2_0xTobs^{2} dif: %e\n",   11./3.*dfdt_0*dfdt_0/f0*Tobs*Tobs*Tobs - gb_max->params[8]);
					
					
	out_file = fopen(argv[5], "w");
	fprintf(out_file, "%.12g %.12g ", logL_max, FF);
	for (j=0; j<gb->NP; j++) fprintf(out_file, "%e ", gb_max->params[j]);
	fclose(out_file);
								
	fprintf(stdout, "\n==============================================================\n");
	
	for(i=0; i<gb->NP; i++) free(Fisher_AE[i]);
	free(XX); free(AA); free(EE);
	
	free(trip->params);
	free(trip);
	
	free(gb->params);
	free(gb);

	free(gb_y->params);
	free(gb_y);
	
	for (c=0; c<NC; c++)
	{
		free(gb_x_arr[c]->params);
		free(gb_x_arr[c]);
	}
	
	free(gb_max->params);
	free(gb_max);
	
	free(XXLS); free(AALS); free(EELS);
	
	return 0;
}

void diff_ev_jump(struct GB *gb_x, struct GB *gb_y, gsl_rng *r, double ***history, long m, int c)
{
	int i, j, k;
		
	double alpha, beta;
	const int NP = gb_x->NP;
	
	if (m/10<2) 
	{
		for (i=0; i<NP; i++) gb_y->params[i] = gb_x->params[i]; // i.e. change nothing
	} else 
	{	
		// select two different points
		j = (int)(((double)m/10.*gsl_rng_uniform(r)));
		do {
			k = (int)(((double)m/10.*gsl_rng_uniform(r)));
		} while (j==k);
		
		alpha = 1.0;
		beta = gsl_rng_uniform(r);
		if (beta < 0.9) alpha = gsl_ran_gaussian(r, 1.);
		
		for (i=0; i<NP; i++)
		{
			gb_y->params[i] = gb_x->params[i] + alpha*(history[c][j][i] - history[c][k][i]);
		}
		
// 		alpha = gsl_ran_gaussian(r, 0.1);
// 		beta = 2.38/sqrt(2.)/9.;
// 		for (i=0; i<NP; i++)
// 		{
// 			gb_y->params[i] = gb_x->params[i] + beta*(history[c][j][i] - history[c][k][i]) + alpha;
// 		}
	}
	
	gb_y->q = (long)(gb_y->params[0]);
	
	return;
}

void prior_jump(struct GB *gb_y, gsl_rng *r)
{
	double f_lo, f_up;
	const int NP = gb_y->NP;
	
	// HACK
	f_lo = 3.0e-3; //2.0e-3; // 7.0e-4;   // lower frequency range for data
	f_up = 3.2e-3; //4.0e-3; // 12.0e-3;  // upper frequency range for data
	
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
	
	return;
}

void fill_data_stream(struct GB *gb, double *XX, double *AA, double *EE, double *XXLS, double *AALS, double *EELS, long NFFT)
{
	const int N = gb->N;
	
	long i, iRe, iIm;
	long k, kRe, kIm;
	
	double sqT = sqrt(Tobs);
		
	for (i=0; i<NFFT; i++) 
	{
		XXLS[i] = 0.;
		AALS[i] = 0.;
		EELS[i] = 0.;
	}
	
	for (i=0; i<N; i++)
	{
		k = (gb->q + i - N/2);   // frequency bin associated with ith sample in GB data arrays
		
		iRe = 2*i;	 kRe = 2*k;
		iIm = 2*i+1; kIm = 2*k+1;
		
		XXLS[kRe] += sqT*XX[iRe];
		XXLS[kIm] += sqT*XX[iIm];
		
		AALS[kRe] += sqT*AA[iRe];
		AALS[kIm] += sqT*AA[iIm];
		
		EELS[kRe] += sqT*EE[iRe];
		EELS[kIm] += sqT*EE[iIm];
	}
	
	return;
}

void print_signal(struct GB *gb, FILE *fptr, double *XX, double *AA, double *EE)
{
	int i, iRe, iIm, k;
	const int N = gb->N;
	
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

double get_overlap(struct GB *gb, double *AA, double *EE, double *AALS, double *EELS)
{
	const int N = gb->N;
	int i, iRe, iIm; 
	int k, kRe, kIm;
	
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

void setup_triple(struct Triple *trip, char *argv[])
{
	double ma, mb, mc, Mc, m2;
	double e2, P2, n2, a2;
	double Ac1c2;
	
	// specify parameters of the triple
	ma	 = 0.3;		                  // Solar mass, mass of body a
	mc   = 0.5;		                  // Solar mass, mass of companion
	Mc   = 0.25;	                  // Solar mass, chirp mass
	e2   = 0.3;		                  // eccentricity of outer binary
	mb   = get_mb(Mc, ma);            // mass of second body in tight binary
	m2   = ma + mb + mc;              // total mass of triple
	P2   = atof(argv[3])*YEAR;        // in years, period of outer orbit
	n2   = PI2/P2;		   			  // outer orbit mean motion
	a2 = pow(G*m2*MSUN/n2/n2, 1./3.); // semi-major axis of outer orbit

	fprintf(stdout, "1/P2: %e\n", 1./P2);
	// Triple
	Ac1c2 = 0.77; // This is the average value
	
	trip->NP = 5;	//  amp2, omegabar, e2, n2, T2
	
	trip->params = malloc(trip->NP*sizeof(double));
	
	trip->params[0] = log(mc*MSUN*sqrt(G/(m2*MSUN)/a2/(1. - e2*e2))*Ac1c2);
	trip->params[1] = 0.; 
	trip->params[2] = e2; 
	trip->params[3] = log(n2*YEAR);
	trip->params[4] = log(atof(argv[4])*P2/YEAR);

	fprintf(stdout, "m2: %e\n", m2);
	
	return;
}

double get_snrAE(struct GB *gb, double *AA, double *EE)
{
	int iRe, iIm;
	int i, k;
	
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

void check_priors(double *params, int *meets_priors, int NP)
{
	double f_lo, f_up;
	
	// HACK
	f_lo = 3.0e-3;//2.0e-3; // 7.0e-4;   // lower frequency range for data
	f_up = 3.2e-3;//4.0e-3; // 12.0e-3;  // upper frequency range for data
 	
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
		if (params[7] < -30.   || params[7] > 30.) 		
		{
			*meets_priors = 0;
			return;
		}     
	}
	if (NP > 8) 
	{
		if (params[8] < -30.   || params[8] > 30.) 
		{
			*meets_priors = 0;
			return;
		}    		   
	}
	
	*meets_priors = 1;
	
	return;
}

double get_logL(struct GB *gb, double *AA, double *EE, double *AALS, double *EELS)
{
	int flag = 0;
	double logL;
	
	if (flag == 0)
	{
		long iRe, iIm, kRe, kIm;
		long i, k;
	
		double f, SnX, SnAE;	
		double sqT;
		double f_lo, f_up;
	
		sqT = sqrt(Tobs);
	
		logL = 0.;
	
		// HACK
		f_lo = 3.0e-3; // 2.0e-3; // 7.0e-4;   // lower frequency range for data
		f_up = 3.2e-3; // 4.0e-3; // 12.0e-3;  // upper frequency range for data

		const int NFFT = (long)(Tobs/dt);
	
		double *a, *e;
		a = malloc(NFFT*sizeof(double));
		e = malloc(NFFT*sizeof(double));
	
		const long i_lo = (long)(f_lo*Tobs);
		const long i_up = (long)(f_up*Tobs);
	
		for (i=i_lo; i<i_up; i++)
		{
			a[2*i] = 0.;
			e[2*i] = 0.;
			
			a[2*i+1] = 0.;
			e[2*i+1] = 0.;
		}
	
		for (i=0; i<gb->N; i++)
		{
			k = (gb->q + i - gb->N/2);    		
		
			iRe = 2*i;	  kRe = 2*k;
			iIm = 2*i+1;  kIm = 2*k+1;
		
			a[kRe] = AA[iRe]*sqT;
			a[kIm] = AA[iIm]*sqT;
		
			e[kRe] = EE[iRe]*sqT;
			e[kIm] = EE[iIm]*sqT;
		}
	
		for (i=i_lo; i<i_up; i++)
		{
			f = (double)(i)/Tobs; 			
			instrument_noise(f, &SnAE, &SnX);
		
			iRe = 2*i;
			iIm = 2*i+1;
		
			logL += ((a[iRe] - AALS[iRe])*(a[iRe] - AALS[iRe])
					+(a[iIm] - AALS[iIm])*(a[iIm] - AALS[iIm])
					+(e[iRe] - EELS[iRe])*(e[iRe] - EELS[iRe])
					+(e[iIm] - EELS[iIm])*(e[iIm] - EELS[iIm]))/SnAE;
			
		}
		logL *= -2.; 
	
		free(a); free(e);
	}
	else logL = -1.;
	
	return logL;
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

double get_f_src(struct GB *gb, double t)
{
	double f;
	double f0, dfdt_0, d2fdt2_0;
	
	dfdt_0   = 0.;
	d2fdt2_0 = 0.;
	
	f0 = gb->params[0]/gb->T;
	if (gb->NP > 7) dfdt_0 = gb->params[7]/gb->T/gb->T;
	if (gb->NP > 8) d2fdt2_0 = 11./3.*dfdt_0*dfdt_0/f0;
	
	f = f0 + dfdt_0*t + 0.5*d2fdt2_0*t*t;
	
	return f;
}		

double get_f0H(struct GB *gb, struct Triple *trip, double t)
{
	double f0H;
	
	f0H  = get_f_src(gb, t); // extract source frame initial frequency
	f0H *= (1. + get_vLOS(trip, t)/C);
	
	return f0H;
}

void get_f0H_derivs(struct GB *gb, struct Triple *trip, double *dfdt_0H, double *d2fdt2_0H)
{
	double ep;	    // epsilon for derivatives

	ep = YEAR/(1.0e6);
		
	*dfdt_0H  = -get_f0H(gb, trip, 2.*ep);
	*dfdt_0H += 8.*get_f0H(gb, trip, ep);
	*dfdt_0H += -8.*get_f0H(gb, trip, -ep);
	*dfdt_0H += get_f0H(gb, trip, -2.*ep);
	*dfdt_0H /= 12.*ep;
	
	ep = YEAR;
	
	*d2fdt2_0H  =   1./90.*get_f0H(gb, trip, -3.*ep);
	*d2fdt2_0H +=  -3./20.*get_f0H(gb, trip, -2.*ep);
	*d2fdt2_0H +=    3./2.*get_f0H(gb, trip, -ep);
	*d2fdt2_0H += -49./18.*get_f0H(gb, trip, 0.);
	*d2fdt2_0H +=    3./2.*get_f0H(gb, trip, ep);
	*d2fdt2_0H +=  -3./20.*get_f0H(gb, trip, 2.*ep);
	*d2fdt2_0H +=   1./90.*get_f0H(gb, trip, 3.*ep);
	*d2fdt2_0H /= ep*ep;
	
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

	gsl_matrix_free (GSLmatrix);
	gsl_matrix_free (GSLinvrse);
	gsl_permutation_free (permutation);

	return cond;
}
