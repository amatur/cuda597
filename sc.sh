#to find factors of a number

for (( i=1; i<=100; i++ ));do
        echo $i;
        mpirun -q -np $i ./jacobi 48;
        #num=$((num/$i))
done > file.txt
sed -i '/MXM/d' ./file.txt
sed -n 1~2p file.txt
sed -n 2~2p file.txt


