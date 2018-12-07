#to find factors of a number
for i in 1 2 4 5 10 16  20  40 80; do

        echo $i;
        ./jacobi 80 $i;
        #num=$((num/$i))
done > file.txt
sed -n 1~2p file.txt
sed -n 2~2p file.txt
