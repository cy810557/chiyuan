#include<iostream>
#include<sstream>
using namespace std;

int atoi(string s){
    int d = 0;
    while(*s.begin() == ' ') s.erase(s.begin());
    stringstream ss;
    ss<<s;
    ss>>d;
    return d;
}

int main(int argc, char *argv[]){
    string str = argv[1];
    cout<<'"'<<str<<'"'<<endl;
    cout<<atoi(str)<<endl;
}
