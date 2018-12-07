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
./jacobi 20 4;
./jacobi 40 2;
./jacobi 80 1;

done > weak.txt

sed -i '/MXM/d' ./weak.txt
#sed -n 3~3p weak.txt
#
#sed -n 2~2p weak.txt
