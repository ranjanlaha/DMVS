# Makefile for C healpix routines 

# HealPix directories
HEALPIX_HOME = /lustre/ki/pfs/dmpowel1/Healpix_3.30

# directories and object/lib versions
SRC = speedyvs.c 
OBJ = $(patsubst %.c, %.o, $(SRC))
SO = $(patsubst %.c, %.so, $(SRC))
INC = #-I$(HEALPIX_HOME)/include 

# compiler options
CC = gcc
CFLAGS += -shared -fPIC -Wall -O3
LDFLAGS += -lm #-L$(HEALPIX_HOME)/lib -l:libchealpix.so

all: $(SO) 

# link and compile to shared object 
$(SO): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LIB) $(LDFLAGS) $(CFLAGS)

# compile
%.o: %.c Makefile 
	$(CC) -c -o $@ $(INC) $(CFLAGS) $< 

clean:
	rm -rf $(OBJ) $(SO) 
