# Makefile for C healpix routines 

# HealPix directories
HEALPIX_HOME = /lustre/ki/pfs/dmpowel1/Healpix_3.30
CFIO_HOME = /lustre/ki/pfs/dmpowel1/cfitsio

# directories and object/lib versions
SRC = modelfit.c sampling.c healpix_f90_wrapper.c nelder-mead/nelder_mead.c 
OBJ = $(patsubst %.c, %.o, $(SRC))
COM = Makefile svs.h GNU_dv.h 
SO = speedyvs.so 

# compiler options
CC = gcc
INC = -I. -Inelder-mead
CFLAGS = -fPIC -shared -Wall -O3 
LDFLAGS = -lm 
LDFLAGS += -L$(HEALPIX_HOME)/libf90 -L$(CFIO_HOME)/lib -lhealpix -lcfitsio -lgfortran

all: $(SO) 

# link and compile to shared object 
$(SO): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LIB) $(LDFLAGS) $(CFLAGS)

# compile
%.o: %.c $(COM) 
	$(CC) -c -o $@ $(INC) $(CFLAGS) $< 

clean:
	rm -rf $(OBJ) $(SO) 
