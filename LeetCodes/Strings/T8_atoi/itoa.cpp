#include<string>
#include<sstream>
#include<iostream>
using namespace std;
int main(){
    int i = 1234;
    string s;
    stringstream ss;
    ss<<i;
    ss>>s;
    cout<<"string: "<<s.substr(0,3)<<", len: "<<s.size()<<endl;
}
