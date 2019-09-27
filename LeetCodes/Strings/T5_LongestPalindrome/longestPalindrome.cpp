#include<iostream>
#include<string>
using namespace std;

int expandString(const string &s, int left, int right){
    int L = left, R = right;
    while(L>=0 && R<s.size() && s[L]==s[R]){
        --L; ++R; //最后多进行了一次！
    }
    cout<<"Current substring(center at "<<left<<", "<<right<<"): ";
    for(int i=L+1;i<=R-1;++i) cout<<s[i];
    cout<<endl;
    return R-L-1;
}

string longestPalindrome(const string &s){
    if (s.size()<=1) return s;
    int len_odd=0, len_even=0;
    int len=0, start=0, end=0;
    for (int i=0;i<s.size();++i){
        len_odd = expandString(s, i, i);
        len_even = expandString(s, i, i+1);
        cout<<endl;
        len = max(len_odd, len_even);
        if(len>(end-start)){
            start = i-(len-1)/2;
            end = i+len/2;
        }
        cout<<len<<" "<<start<<" "<<end<<endl;
    }
    return s.substr(start, end-start+1);
}

int main(int argc, char *argv[]){
    if (argc>1){
        cout<<"input: "<<argv[1]<<endl;
        cout<<"result"<<longestPalindrome(argv[1])<<endl;
    } else {
        string str = "b";
        cout<<"input: "<<str<<endl;
        auto rst = longestPalindrome(str);
        cout<<"result: "<<rst<<endl;
    }
    return 0;
}
