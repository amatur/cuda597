# Makefile for parallelization of jacobi
# to use this, modules:
# module use /storage/work/a/awl5173/toShare/tauPdt/tau
# module load adamsTau_2.27


## uncomment for parallel without tau
#CXX=mpic++
CC=gcc
CXX=mpic++
#CXX=tau_cxx.sh
#CC=tau_cc.sh

LIBS=-lm

CCFLAGS= -g -O3

#~ CCFLAGS=-Wall -O3


#uncomment this line for Serial only
#all: jacobi_s jacobi_s_opt

#uncomment this line for Parallel only
#all: jacobi jacobi_p_unopt

all: lu
#all: jacobi jacobi_p_unopt jacobi_s jacobi_s_opt

data_parallel:
	for i in 10;\
	do	\
		mpirun -np $$i ./jacobi 10; \
	done

data_serial:
	echo -e '"Dimension"' " " '"Time (sec)"' ;
	for i in 5 10 20 30 40 50 ;	\
	do	\
	 ./jacobi_s $$i; \
	done

data_serial_opt:
	echo -e '"Dimension"' " " '"Time (sec)"' ;
	for i in 5 10 20 30 40 50 ;	\
	do	\
	 ./jacobi_s_opt $$i; \
	done

run:
	echo "### RUN PARALLEL:"; mpirun -np 10 ./jacobi 40 \
	echo "RUN SERIAL"; ./jacobi_s_opt 40 \


runp:
	echo "## RUNNING PARALLEL JACOBI"; mpirun -np 10 ./jacobi 40 \

runs:
	echo "## RUNNING SERIAL JACOBI"; ./jacobi_s_opt 40 \


runpu:
	echo "## RUNNING PARALLEL JACOBI *Unoptimized*"; mpirun -np 10 ./jacobi_p_unopt 40 \

runsu:
	echo "## RUNNING SERIAL JACOBI *Unoptimized*"; ./jacobi_s 10 \

#~ test:
#~ 	echo "TESTING";\
#~ 	./test

jacobi: jacobi.o matrix_util.o
	$(CXX) $(CCFLAGS) -o $@ $^ $(LIBS)

jacobi_s: jacobi_s.o matrix_util.o
	$(CC) $(CCFLAGS) -o $@ $^

jacobi_s.o: jacobi_s.cpp jacobi.h
	$(CC) $(CCFLAGS) -c $<

jacobi_s_opt: jacobi_s_opt.o matrix_util.o
	$(CC) $(CCFLAGS) -o $@ $^

jacobi_s_opt.o: jacobi_s_opt.cpp jacobi.h
	$(CC) $(CCFLAGS) -c $<


jacobi_p_unopt: jacobi_p_unopt.o matrix_util.o
	$(CXX) $(CCFLAGS) -o $@ $^

jacobi_p_unopt.o: jacobi_p_unopt.cpp jacobi.h
	$(CXX) $(CCFLAGS) -c $<



lu: lu.o matrix_util.o
	$(CXX) $(CCFLAGS) -o $@ $^

test: test.o jacobi.o lu.o matrix_util.o
	$(CXX) $(CCFLAGS) -o $@ $^

%.o: %.cpp %.h
	$(CXX) $(CCFLAGS) -c $<

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $<

clean:
	rm -f *.o jacobi lu test jacobi_s jacobi_s_opt jacobi_p_unopt profile.*
