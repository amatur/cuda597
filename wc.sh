#to find factors of a number
#
#for (( i=1; i<=64; i=i*2 ));do
#        echo $i;
#        x=$((i*2));
#        calc=$(echo "sqrt ( $x  )" | bc -l) ;        
#        echo $calc;
#        echo 1 $calc | awk '{ r=$1 % $2; q=y; if (r != 0) q=int(q+1); print q}'
#        #mpirun -q -np $i ./jacobi $((i*2));
#        #num=$((num/$i))
#done>weak.txt
for i in 1; do
mpirun -q -np 2 ./jacobi 4  
mpirun -q -np 4 ./jacobi 6 
mpirun -q -np 8 ./jacobi 8 
mpirun -q -np 16 ./jacobi 12 
mpirun -q -np 32 ./jacobi 16 
mpirun -q -np 64 ./jacobi 22 
done > weak.txt

sed -i '/MXM/d' ./weak.txt
#sed -n 3~3p weak.txt
#
#sed -n 2~2p weak.txt 
