#include "scc.h"
#include<stack>
#include<unistd.h>
#include "load.h"
#include "scc_kernels.h"
using namespace std;

void print_help(){
	printf("To execute:\n");
	printf("./scc -a [g/h/x/y/d] -p [0/1] -q [0/1] -w [1/2/4/8/16/32] -f <file>\n");
	printf("-a algorith to use   g - vHong, h - vSlota, x - wHong, y - wSlota, d - detectSCC\n");
	printf("-p Trim-1 enable\n");
	printf("-q Trim-2 enable\n");
	printf("-w warp size\n");
	printf("-f input file\n");
    	printf("Please note that the input graph must be in GTgraph format\n");
    	printf("If not provided the results are undefined\n");
	return;
}

int main( int argc, char** argv ){

    if ( argc < 11 ) {
	print_help();
	return 1;
    }

    char *file = NULL;
    char c, algo;
    bool trim1 = true, trim2 = true;
    int warpSize = 1;

    while((c = getopt(argc, argv, "a:p:q:w:f:")) != -1){
        switch(c){
            case 'a':
                algo = optarg[0];
                break;

            case 'p':
                trim1 = optarg[0]=='0'?false:true;
                break;

            case 'q':
                trim2 = optarg[0]=='0'?false:true;
                break;

            case 'w':
                warpSize = atoi(optarg);
                break;

            case 'f':
                file = optarg;
		break;

		default:
			print_help();
			return 1;
        }
    }

    // CSR representation
    uint32_t CSize; // column arrays size
    uint32_t RSize; // range arrays size
    // Forwards arrays
    uint32_t *Fc = NULL; // forward columns
    uint32_t *Fr = NULL; // forward ranges
    // Backwards arrays
    uint32_t *Bc = NULL; // backward columns
    uint32_t *Br = NULL; // backward ranges

    uint32_t *Pr = NULL;

    //obtain a CSR graph representation
    loadFullGraph(file, &CSize, &RSize, &Fc, &Fr, &Bc, &Br, &Pr);

    try {

        switch(algo){
            case 'g':
                vHong( CSize, RSize, Fc, Fr, Bc, Br, trim1, trim2);
                break;

            case 'h':
                vSlota( CSize, RSize, Fc, Fr, Bc, Br, trim1, trim2);
                break;

            case 'x':
                wHong( CSize, RSize, Fc, Fr, Bc, Br, trim1, trim2, warpSize);
                break;

            case 'y':
                wSlota( CSize, RSize, Fc, Fr, Bc, Br, trim1, trim2, warpSize);
                break;

            case 'd':
                detectSCC( CSize, RSize, Fc, Fr, Bc, Br, Pr, trim1, trim2);
				break;

		default:
			print_help();
			return 1;
        }
    }
    catch (const char * e)
    {
        printf("%s\n",e);
        return 1;
    }
	printf("\n");
    delete [] Fr;
    delete [] Fc;
    delete [] Br;
    delete [] Bc;
    return 0;
}
