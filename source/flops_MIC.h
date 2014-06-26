/*
  based on flops_AVX.h 
 */

#ifndef _MIC_h
#define _MIC_h
#include <immintrin.h>

#include "flops.h"

#define AND(a,b)    _mm512_castsi512_pd(_mm512_and_epi32 (_mm512_castpd_si512(a), _mm512_castpd_si512(b)))
#define OR(a,b)     _mm512_castsi512_pd(_mm512_or_epi32 (_mm512_castpd_si512(a), _mm512_castpd_si512(b)))
#define XOR(a,b)    _mm512_castsi512_pd(_mm512_xor_epi32(_mm512_castpd_si512(a), _mm512_castpd_si512(b)))
#define ANDNOT(a,b) _mm512_castsi512_pd(_mm512_andnot_epi32 (_mm512_castpd_si512(a), _mm512_castpd_si512(b)))

double test_dp_mul_MIC_internal(double x,double y,size_t iterations){
    register __m512d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB;

    r0 = _mm512_set1_pd(x);
    r1 = _mm512_set1_pd(y);

    r8 = _mm512_set1_pd(-0.0);

    r2 = XOR(r0,r8);
    r3 = OR(r0,r8);
    r4 = ANDNOT(r8,r0);

    r5 = _mm512_mul_pd(r1,_mm512_set1_pd(0.37796447300922722721));
    r6 = _mm512_mul_pd(r1,_mm512_set1_pd(0.24253562503633297352));
    r7 = _mm512_mul_pd(r1,_mm512_set1_pd(4.1231056256176605498));
    r8 = _mm512_add_pd(r0,_mm512_set1_pd(2.3));
    r9 = _mm512_sub_pd(r1,_mm512_set1_pd(2.3));

//    r8 = _mm512_set1_pd(1.4142135623730950488);
//    r9 = _mm512_set1_pd(1.7320508075688772935);
//    rA = _mm512_set1_pd(0.57735026918962576451);
//    rB = _mm512_set1_pd(0.70710678118654752440);

    rA = _mm512_set1_pd(1.4142135623730950488);
    rB = _mm512_set1_pd(0.70710678118654752440);

    uint64 iMASK = 0x800fffffffffffffull;
    __m512d MASK = _mm512_set1_pd(*(double*)&iMASK);
    __m512d vONE = _mm512_set1_pd(1.0);

    size_t c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){
            r0 = _mm512_mul_pd(r0,rA);
            r1 = _mm512_mul_pd(r1,rA);
            r2 = _mm512_mul_pd(r2,rA);
            r3 = _mm512_mul_pd(r3,rA);
            r4 = _mm512_mul_pd(r4,rA);
            r5 = _mm512_mul_pd(r5,rA);
            r6 = _mm512_mul_pd(r6,rA);
            r7 = _mm512_mul_pd(r7,rA);
            r8 = _mm512_mul_pd(r8,rA);
            r9 = _mm512_mul_pd(r9,rA);

            r0 = _mm512_mul_pd(r0,rB);
            r1 = _mm512_mul_pd(r1,rB);
            r2 = _mm512_mul_pd(r2,rB);
            r3 = _mm512_mul_pd(r3,rB);
            r4 = _mm512_mul_pd(r4,rB);
            r5 = _mm512_mul_pd(r5,rB);
            r6 = _mm512_mul_pd(r6,rB);
            r7 = _mm512_mul_pd(r7,rB);
            r8 = _mm512_mul_pd(r8,rB);
            r9 = _mm512_mul_pd(r9,rB);

            i++;
        }

        //print(r0);
        //print(r1);
        //print(r2);
        //print(r3);
        //print(r4);
        //print(r5);
        //print(r6);
        //print(r7);
        //cout << endl;
        
        r0 = AND(r0,MASK);
        r1 = AND(r1,MASK);
        r2 = AND(r2,MASK);
        r3 = AND(r3,MASK);
        r4 = AND(r4,MASK);
        r5 = AND(r5,MASK);
        r6 = AND(r6,MASK);
        r7 = AND(r7,MASK);
        r8 = AND(r8,MASK);
        r9 = AND(r9,MASK);
        r0 = OR(r0,vONE);
        r1 = OR(r1,vONE);
        r2 = OR(r2,vONE);
        r3 = OR(r3,vONE);
        r4 = OR(r4,vONE);
        r5 = OR(r5,vONE);
        r6 = OR(r6,vONE);
        r7 = OR(r7,vONE);
        r8 = OR(r8,vONE);
        r9 = OR(r9,vONE);

        c++;
    }

//    wclk end = wclk_now();
//    double secs = wclk_secs_since(start);
//    uint64 ops = 12 * 1000 * c * 2;
//    cout << "Seconds = " << secs << endl;
//    cout << "FP Ops  = " << ops << endl;
//    cout << "FLOPs   = " << ops / secs << endl;
    
    r0 = _mm512_add_pd(r0,r1);
    r2 = _mm512_add_pd(r2,r3);
    r4 = _mm512_add_pd(r4,r5);
    r6 = _mm512_add_pd(r6,r7);
    r8 = _mm512_add_pd(r8,r9);
    
    r0 = _mm512_add_pd(r0,r2);
    r4 = _mm512_add_pd(r4,r6);

    r0 = _mm512_add_pd(r0,r4);
    r0 = _mm512_add_pd(r0,r8);

    double out = 0;
    __m512d tmp = r0;
    out += ((double*)&tmp)[0];
    out += ((double*)&tmp)[1];
    out += ((double*)&tmp)[2];
    out += ((double*)&tmp)[3];
    out += ((double*)&tmp)[4];
    out += ((double*)&tmp)[5];
    out += ((double*)&tmp)[6];
    out += ((double*)&tmp)[7];

    return out;
}
void test_dp_mul_MIC(int tds,size_t iterations){
    
    printf("Testing MIC Mul:\n");
    double *sum = (double*)malloc(tds * sizeof(double));
    wclk start = wclk_now();
    
#pragma omp parallel num_threads(tds)
    {
        double ret = test_dp_mul_MIC_internal(1.1,2.1,iterations);
        sum[omp_get_thread_num()] = ret;
    }

    double secs = wclk_secs_since(start);
    uint64 ops = 20 * 1000 * iterations * tds * 8;
    printf("Seconds = %g\n",secs);
    printf("FP Ops  = %llu\n",(unsigned long long)ops);
    printf("FLOPs   = %g\n",ops / secs / 1.0e9);

    double out = 0;
    int c = 0;
    while (c < tds){
        out += sum[c++];
    }
    
    printf("sum = %g\n\n",out);

    free(sum);
}

double test_dp_add_MIC_internal(double x,double y,size_t iterations){
    register __m512d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB;

    r0 = _mm512_set1_pd(x);
    r1 = _mm512_set1_pd(y);

    r8 = _mm512_set1_pd(-0.0);
    r9 = _mm512_set1_pd(0.5);

    r2 = XOR(r0,r8);
    r3 = OR(r0,r8);
    r4 = ANDNOT(r8,r0);
    r5 = _mm512_mul_pd(r1,r9);
    r6 = _mm512_add_pd(r1,r9);
    r7 = _mm512_sub_pd(r1,r9);
    r8 = _mm512_add_pd(r0,_mm512_set1_pd(2.3));
    r9 = _mm512_sub_pd(r1,_mm512_set1_pd(2.3));

    uint64 iMASK = 0x800fffffffffffffull;
    __m512d MASK = _mm512_set1_pd(*(double*)&iMASK);
    __m512d vONE = _mm512_set1_pd(1.0);

    rA = _mm512_set1_pd(0.1);
    rB = _mm512_set1_pd(0.1001);

//    wclk start = wclk_now();
    size_t c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){
            r0 = _mm512_add_pd(r0,rA);
            r1 = _mm512_add_pd(r1,rA);
            r2 = _mm512_add_pd(r2,rA);
            r3 = _mm512_add_pd(r3,rA);
            r4 = _mm512_add_pd(r4,rA);
            r5 = _mm512_add_pd(r5,rA);
            r6 = _mm512_add_pd(r6,rA);
            r7 = _mm512_add_pd(r7,rA);
            r8 = _mm512_add_pd(r8,rA);
            r9 = _mm512_add_pd(r9,rA);

            r0 = _mm512_sub_pd(r0,rB);
            r1 = _mm512_sub_pd(r1,rB);
            r2 = _mm512_sub_pd(r2,rB);
            r3 = _mm512_sub_pd(r3,rB);
            r4 = _mm512_sub_pd(r4,rB);
            r5 = _mm512_sub_pd(r5,rB);
            r6 = _mm512_sub_pd(r6,rB);
            r7 = _mm512_sub_pd(r7,rB);
            r8 = _mm512_sub_pd(r8,rB);
            r9 = _mm512_sub_pd(r9,rB);

            //r8 = _mm512_add_pd(r0,r1);
            //r9 = _mm512_add_pd(r2,r3);
            //rA = _mm512_add_pd(r4,r5);
            //rB = _mm512_add_pd(r6,r7);

            //r0 = _mm512_sub_pd(r0,r4);
            //r1 = _mm512_sub_pd(r1,r5);
            //r2 = _mm512_sub_pd(r2,r6);
            //r3 = _mm512_sub_pd(r3,r7);

            //r4 = _mm512_add_pd(r4,r8);
            //r5 = _mm512_add_pd(r5,r9);
            //r6 = _mm512_add_pd(r6,rA);
            //r7 = _mm512_add_pd(r7,rB);

            i++;
        }

        //print(r0);
        //print(r1);
        //print(r2);
        //print(r3);
        //print(r4);
        //print(r5);
        //print(r6);
        //print(r7);
        //cout << endl;

        r0 = AND(r0,MASK);
        r1 = AND(r1,MASK);
        r2 = AND(r2,MASK);
        r3 = AND(r3,MASK);
        r4 = AND(r4,MASK);
        r5 = AND(r5,MASK);
        r6 = AND(r6,MASK);
        r7 = AND(r7,MASK);
        r8 = AND(r8,MASK);
        r9 = AND(r9,MASK);
        r0 = OR(r0,vONE);
        r1 = OR(r1,vONE);
        r2 = OR(r2,vONE);
        r3 = OR(r3,vONE);
        r4 = OR(r4,vONE);
        r5 = OR(r5,vONE);
        r6 = OR(r6,vONE);
        r7 = OR(r7,vONE);
        r8 = OR(r8,vONE);
        r9 = OR(r9,vONE);

        c++;
    }

//    wclk end = wclk_now();
//    double secs = wclk_secs_since(start);
//    uint64 ops = 12 * 1000 * c * 4;
//    cout << "Seconds = " << secs << endl;
//    cout << "FP Ops  = " << ops << endl;
//    cout << "FLOPs   = " << ops / secs << endl;

    r0 = _mm512_add_pd(r0,r1);
    r2 = _mm512_add_pd(r2,r3);
    r4 = _mm512_add_pd(r4,r5);
    r6 = _mm512_add_pd(r6,r7);
    r8 = _mm512_add_pd(r8,r9);
    
    r0 = _mm512_add_pd(r0,r2);
    r4 = _mm512_add_pd(r4,r6);

    r0 = _mm512_add_pd(r0,r4);
    r0 = _mm512_add_pd(r0,r8);

    double out = 0;
    __m512d tmp = r0;
    out += ((double*)&tmp)[0];
    out += ((double*)&tmp)[1];
    out += ((double*)&tmp)[2];
    out += ((double*)&tmp)[3];
    out += ((double*)&tmp)[4];
    out += ((double*)&tmp)[5];
    out += ((double*)&tmp)[6];
    out += ((double*)&tmp)[7];

    return out;
}

void test_dp_add_MIC(int tds,size_t iterations){
    
    printf("Testing MIC Add:\n");
    double *sum = (double*)malloc(tds * sizeof(double));
    wclk start = wclk_now();
    
#pragma omp parallel num_threads(tds)
    {
        double ret = test_dp_add_MIC_internal(1.1,2.1,iterations);
        sum[omp_get_thread_num()] = ret;
    }

    double secs = wclk_secs_since(start);
    uint64 ops = 20 * 1000 * iterations * tds * 8;
    printf("Seconds = %g\n",secs);
    printf("FP Ops  = %llu\n",(unsigned long long)ops);
    printf("GFLOPs   = %g\n",ops / secs / 1.0e9);

    double out = 0;
    int c = 0;
    while (c < tds){
        out += sum[c++];
    }
    
    printf("sum = %g\n\n",out);

    free(sum);
}

double test_dp_fma_MIC_internal(double x,double y,size_t iterations){
    register __m512d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    r0 = _mm512_set1_pd(x);
    r1 = _mm512_set1_pd(y);
    r8 = _mm512_set1_pd(-0.0);
    //    r2 =  _mm512_xor_pd(r0,r8));
    //    r3 = _mm512_or_pd(r0,r8);
    //    r4 = _mm512_andnot_pd(r8,r0);
    r2 = XOR(r0,r8);
    r3 = OR(r0,r8);
    r4 = ANDNOT(r0,r8);
    r5 = _mm512_mul_pd(r1,_mm512_set1_pd(0.37796447300922722721));
    r6 = _mm512_mul_pd(r1,_mm512_set1_pd(0.24253562503633297352));
    r7 = _mm512_mul_pd(r1,_mm512_set1_pd(4.1231056256176605498));
    r8 = _mm512_add_pd(r0,_mm512_set1_pd(0.37796447300922722721));
    r9 = _mm512_add_pd(r1,_mm512_set1_pd(0.24253562503633297352));
    rA = _mm512_sub_pd(r0,_mm512_set1_pd(4.1231056256176605498));
    rB = _mm512_sub_pd(r1,_mm512_set1_pd(4.1231056256176605498));
    rC = _mm512_set1_pd(1.0488088481701515470);
    rD = _mm512_set1_pd(0.95346258924559231545);
    rE = _mm512_set1_pd(1.1);
    rF = _mm512_set1_pd(0.90909090909090909091);

    uint64 iMASK = 0x800fffffffffffffull;
    __m512d MASK = _mm512_set1_pd(*(double*)&iMASK);
    __m512d vONE = _mm512_set1_pd(1.0);

    size_t c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < 1000){
            r0 = _mm512_fmadd_pd(r0,rC,rD);
            r1 = _mm512_fmadd_pd(r1,rC,rD);
            r2 = _mm512_fmadd_pd(r2,rC,rD);
            r3 = _mm512_fmadd_pd(r3,rC,rD);
            r4 = _mm512_fmadd_pd(r3,rC,rD);
            r5 = _mm512_fmadd_pd(r4,rC,rD);
            r6 = _mm512_fmadd_pd(r5,rC,rD);
            r7 = _mm512_fmadd_pd(r7,rC,rD);
            r8 = _mm512_fmadd_pd(r8,rC,rD);
            r9 = _mm512_fmadd_pd(r9,rC,rD);
            rA = _mm512_fmadd_pd(rA,rC,rD);
            rB = _mm512_fmadd_pd(rB,rC,rD);

            r0 = _mm512_fmadd_pd(r0,rD,rF);
            r1 = _mm512_fmadd_pd(r1,rD,rF);
            r2 = _mm512_fmadd_pd(r2,rD,rF);
            r3 = _mm512_fmadd_pd(r3,rD,rF);
            r4 = _mm512_fmadd_pd(r4,rD,rF);
            r5 = _mm512_fmadd_pd(r5,rD,rF);
            r6 = _mm512_fmadd_pd(r6,rD,rF);
            r7 = _mm512_fmadd_pd(r7,rD,rF);
            r8 = _mm512_fmadd_pd(r8,rD,rF);
            r9 = _mm512_fmadd_pd(r9,rD,rF);
            rA = _mm512_fmadd_pd(rA,rD,rF);
            rB = _mm512_fmadd_pd(rB,rD,rF);

            r0 = _mm512_fmadd_pd(r0,rC,rD);
            r1 = _mm512_fmadd_pd(r1,rC,rD);
            r2 = _mm512_fmadd_pd(r2,rC,rD);
            r3 = _mm512_fmadd_pd(r3,rC,rD);
            r4 = _mm512_fmadd_pd(r3,rC,rD);
            r5 = _mm512_fmadd_pd(r4,rC,rD);
            r6 = _mm512_fmadd_pd(r5,rC,rD);
            r7 = _mm512_fmadd_pd(r7,rC,rD);
            r8 = _mm512_fmadd_pd(r8,rC,rD);
            r9 = _mm512_fmadd_pd(r9,rC,rD);
            rA = _mm512_fmadd_pd(rA,rC,rD);
            rB = _mm512_fmadd_pd(rB,rC,rD);

            r0 = _mm512_fmadd_pd(r0,rD,rF);
            r1 = _mm512_fmadd_pd(r1,rD,rF);
            r2 = _mm512_fmadd_pd(r2,rD,rF);
            r3 = _mm512_fmadd_pd(r3,rD,rF);
            r4 = _mm512_fmadd_pd(r4,rD,rF);
            r5 = _mm512_fmadd_pd(r5,rD,rF);
            r6 = _mm512_fmadd_pd(r6,rD,rF);
            r7 = _mm512_fmadd_pd(r7,rD,rF);
            r8 = _mm512_fmadd_pd(r8,rD,rF);
            r9 = _mm512_fmadd_pd(r9,rD,rF);
            rA = _mm512_fmadd_pd(rA,rD,rF);
            rB = _mm512_fmadd_pd(rB,rD,rF);

            i++;
        }

        r0 = AND(r0,MASK);
        r1 = AND(r1,MASK);
        r2 = AND(r2,MASK);
        r3 = AND(r3,MASK);
        r4 = AND(r4,MASK);
        r5 = AND(r5,MASK);
        r6 = AND(r6,MASK);
        r7 = AND(r7,MASK);
        r8 = AND(r8,MASK);
        r9 = AND(r9,MASK);
        rA = AND(rA,MASK);
        rB = AND(rB,MASK);
        r0 = OR(r0,vONE);
        r1 = OR(r1,vONE);
        r2 = OR(r2,vONE);
        r3 = OR(r3,vONE);
        r4 = OR(r4,vONE);
        r5 = OR(r5,vONE);
        r6 = OR(r6,vONE);
        r7 = OR(r7,vONE);
        r8 = OR(r8,vONE);
        r9 = OR(r9,vONE);
        rA = OR(rA,vONE);
        rB = OR(rB,vONE);

	/*
        r0 = _mm512_and_pd(r0,MASK);
        r1 = _mm512_and_pd(r1,MASK);
        r2 = _mm512_and_pd(r2,MASK);
        r3 = _mm512_and_pd(r3,MASK);
        r4 = _mm512_and_pd(r4,MASK);
        r5 = _mm512_and_pd(r5,MASK);
        r6 = _mm512_and_pd(r6,MASK);
        r7 = _mm512_and_pd(r7,MASK);
        r8 = _mm512_and_pd(r8,MASK);
        r9 = _mm512_and_pd(r9,MASK);
        rA = _mm512_and_pd(rA,MASK);
        rB = _mm512_and_pd(rB,MASK);
        r0 = _mm512_or_p(dr0,vONE);
        r1 = _mm512_or_pd(r1,vONE);
        r2 = _mm512_or_pd(r2,vONE);
        r3 = _mm512_or_pd(r3,vONE);
        r4 = _mm512_or_pd(r4,vONE);
        r5 = _mm512_or_pd(r5,vONE);
        r6 = _mm512_or_pd(r6,vONE);
        r7 = _mm512_or_pd(r7,vONE);
        r8 = _mm512_or_pd(r8,vONE);
        r9 = _mm512_or_pd(r9,vONE);
        rA = _mm512_or_pd(rA,vONE);
        rB = _mm512_or_pd(rB,vONE);
	*/

        c++;
    }

//    wclk end = wclk_now();
//    double secs = wclk_secs_since(start);
//    uint64 ops = 12 * 1000 * c * 2;
//    cout << "Seconds = " << secs << endl;
//    cout << "FP Ops  = " << ops << endl;
//    cout << "FLOPs   = " << ops / secs << endl;

    r0 = _mm512_add_pd(r0,r1);
    r2 = _mm512_add_pd(r2,r3);
    r4 = _mm512_add_pd(r4,r5);
    r6 = _mm512_add_pd(r6,r7);
    r8 = _mm512_add_pd(r8,r9);
    rA = _mm512_add_pd(rA,rB);
    
    r0 = _mm512_add_pd(r0,r2);
    r4 = _mm512_add_pd(r4,r6);
    r8 = _mm512_add_pd(r8,rA);

    r0 = _mm512_add_pd(r0,r4);
    r0 = _mm512_add_pd(r0,r8);

    double out = 0;
    __m512d tmp = r0;
    out += ((double*)&tmp)[0];
    out += ((double*)&tmp)[1];
    out += ((double*)&tmp)[2];
    out += ((double*)&tmp)[3];
    out += ((double*)&tmp)[4];
    out += ((double*)&tmp)[5];
    out += ((double*)&tmp)[6];
    out += ((double*)&tmp)[7];

    return out;
}

void test_dp_fma_MIC(int tds,size_t iterations){
    
    printf("Testing MIC FMA:\n");
    double *sum = (double*)malloc(tds * sizeof(double));
    wclk start = wclk_now();
    
#pragma omp parallel num_threads(tds)
    {
        double ret = test_dp_fma_MIC_internal(1.1,2.1,iterations);
        sum[omp_get_thread_num()] = ret;
    }

    double secs = wclk_secs_since(start);
    uint64 ops = 12*4*2 * 1000 * iterations * tds * 8;
    printf("Seconds = %g\n",secs);
    printf("FP Ops  = %llu\n",(unsigned long long)ops);
    printf("GFLOPs   = %g\n",ops / secs / 1.0e9);

    double out = 0;
    int c = 0;
    while (c < tds){
        out += sum[c++];
    }
    
    printf("sum = %g\n\n", out);

    free(sum);
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#endif
