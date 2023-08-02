gcc qaoa_fur.c fur.c diagonal.c -I. -fopenmp -shared -fPIC -Wall -Wl,--version-script=libcsim.map -o ../libcsim.so
