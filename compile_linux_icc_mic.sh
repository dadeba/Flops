#icc -mmic source/*.c -o "flops_SSE2_gcc.out" -lm -fopenmp -msse2 -O2 -D "x86_SSE2" #
#icc -mmic source/*.c -o "flops_AVX_gcc.out"  -lm -fopenmp -mavx  -O2 -D "x86_AVX"  #
#icc source/*.c -o "flops_FMA4_gcc.out" -lm -fopenmp -mfma4 -O2 -D "x86_FMA4"  #

icc source/*.c -o "flops_AVX_icc.out" -mavx -lm -fopenmp -O2 -D "x86_AVX"
icc -mmic source/*.c -o "flops_MIC_icc.out" -mmic -lm -fopenmp -O2 -D "x86_MIC"
icc -mmic source/main.c -o mic.s -mmic -lm -fopenmp -O2 -D "x86_MIC" -S
