# Makefile for parallelization of jacobi
# to use this, modules:
# module use /storage/work/a/awl5173/toShare/tauPdt/tau
# module load adamsTau_2.27


## uncomment for parallel without tau
#CXX=mpic++
CC=nvcc
CXX=gcc

#CXX=tau_cxx.sh
#CC=tau_cc.sh

LIBS=-lm
LDFLAGS=-lcublas -lcurand
CCFLAGS=-g -O3
CUDAFLAGS = -arch=sm_35 -m64 -O3 --use_fast_math
#~ CCFLAGS=-Wall -O3


#uncomment this line for Serial only
#all: jacobi_s jacobi_s_opt

#uncomment this line for Parallel only
#all: jacobi jacobi_p_unopt

all: jacobicu jacobith jacobicublas
	#jacobirand
#all: jacobi jacobi_p_unopt jacobi_s jacobi_s_opt




#-arch=compute_35 -code=sm_35
jacobicu: jacobi.cu
	$(CC) -o jacobi $(CUDAFLAGS) jacobi.cu $(LDFLAGS)

jacobith: jacobi_thrust.cu
	$(CC) -o jacobi_thrust $(CUDAFLAGS) jacobi_thrust.cu $(LDFLAGS)


jacobirand: jacobi_rand.cu
	$(CC) -o jacobi_rand $(CUDAFLAGS) jacobi_rand.cu $(LDFLAGS)

jacobicublas: jacobi_cublas.cu
		$(CC) -o jacobi_cublas $(CUDAFLAGS) jacobi_cublas.cu $(LDFLAGS)


runcu:
	echo "## RUNNING OPTIMIZED CUDA JACOBI"; ./jacobi 16 4\

runth:
		echo "## RUNNING CUDA JACOBI WITH THRUST (Unoptimized)"; ./jacobi_thrust 16 4\

runcublas:
		echo "## RUNNING CUDA JACOBI WITH CUBLAS"; ./jacobi_cublas 16 4\

# data_parallel:
# 	for i in 10;\
# 	do	\
# 		mpirun -np $$i ./jacobi 10; \
# 	done

data:
	echo -e '"Dimension"' " " '"Time (sec)"' ;
	for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ;	\
	do	\
		echo $$i; \
	 ./jacobi $$i $$i; \
	done

data2:
	echo -e '"Dimension"' " " '"Time (sec)"' ;
	for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ;	\
	do	\
		echo $$i; \
	 ./jacobi_cublas $$i $$i; \
	done


%.o: %.cpp %.h
	$(CXX) $(CCFLAGS) -c $<

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $<

clean:
	rm -f *.o jacobi  jacobi_rand jacobi_cublas jacobi_thrust
