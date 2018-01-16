CC = gcc-7

LIBS  = gsl gslcblas m 
CCFLAGS += -g -Wall -std=gnu99 -fmax-errors=5 -O2 -ffast-math -ftree-vectorize 
#CCFLAGS += -Werror 

OBJS = GB.o LISA.o Triple.o

all : $(OBJS) gb_test sim_ann


LISA.o : LISA.c LISA.h
	$(CC) $(CCFLAGS) -c LISA.c

GB.o : GB.c GB.h LISA.h Constants.h 
	$(CC) $(CCFLAGS) -c GB.c 

Triple.o : Triple.c Triple.h LISA.h Constants.h 
	$(CC) $(CCFLAGS) -c Triple.c 	
	
gb_test : $(OBJS) gb_test.c 
	$(CC) $(CCFLAGS) -o gb_test gb_test.c $(OBJS) $(INCDIR:%=-I%) $(LIBDIR:%=-L%) $(LIBS:%=-l%)
	
sim_ann : $(OBJS) Sim_Anneal.c 
	$(CC) $(CCFLAGS) -o sim_ann Sim_Anneal.c $(OBJS) $(INCDIR:%=-I%) $(LIBDIR:%=-L%) $(LIBS:%=-l%)

	
clean: 
	rm *.o gb_test sim_ann