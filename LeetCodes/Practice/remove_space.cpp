#include<iostream>
using namespace std;

char * removeSpace(char* s){
    char* p = s;
    char* q = s;
    while(*q != '\0'){
        if(isspace(*q)) ++q;
        *p++ = *q++;
    }
    *p = '\0';
    return s;
}

int main(int argc, char* argv[]){
    cout<<argv[1]<<endl;
    cout<<removeSpace(argv[1])<<endl;
}

