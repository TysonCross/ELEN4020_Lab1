// Tyson Cross 1239448
// Michael Nortje 1389486 
// Josh Isserow 675720

#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
int main() {
#pragma omp parallel
    {
        printf("hello world \n");
    }
}

