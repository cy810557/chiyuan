# 涉及一个awk中使用shell变量的问题。见https://stackoverflow.com/questions/19075671/how-do-i-use-shell-variables-in-an-awk-script
#ls $1 | awk '/xml/{print "cat ${1}/"$0" | grep /name"}' | bash | awk -F '>|<' '{print $3}' | uniq

#parDir=`dirname $1`
ls $1 | awk -v var=$1 '/xml/{print "cat "var"/"$0" | grep /name"}' | bash | awk -F '>|<' '{print $3}' | uniq | tr "\n" " " > $1/ClassNames.txt 
