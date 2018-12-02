echo "Serial version"
for (( i=1; i<=80; i++ ));do
        echo $i;
        ./jacobi_s_opt $i;
        #num=$((num/$i))
done > file.txt
sed -i '/MXM/d' ./file.txt
sed -n 1~2p file.txt
sed -n 2~2p file.txt


