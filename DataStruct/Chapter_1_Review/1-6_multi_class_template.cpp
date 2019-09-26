#include<iostream>
#include<string>
using namespace std;
template<class Ta, class Tb, class Tc>
Ta abc(const Ta& a, const Tb& b, const Tc& c){
    return a + b * c;
}

int main(){
    string a = "string a ";
    int b = 10;
    float c = -9.999;
    cout<<"result: "<<abc(a, b, c)<<endl;
    
}
