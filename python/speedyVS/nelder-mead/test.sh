#! /bin/bash

gcc -o nm main.c nelder_mead.c 

time ./nm -2.10 -3.04 4.50 
