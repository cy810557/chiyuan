#include<iostream>
#include<vector>
using namespace std;

bool contain(const string &s1, const string& s2){
    for (auto x:s2){
        if (s1.find(x)==string::npos) return false;
    }
    return true;
}

string minWindow(string S, string T){
    string window;
    string result(S);
    int left = 0;
    bool flag = false;
    for (int i=0;i<S.size();++i){
        window.insert(window.end(), S[i]);
        while(contain(window, T)){
            flag = true;
            window.erase(window.begin());
            result = (window.size()+1>=result.size() ? result:S[left]+window);
            ++left;
        }
        /**
        cout<<"Window: ";
        for (auto x : window) cout<<x<<", ";
        cout<<endl;
        cout<<"Result: ";
        for (auto x : result) cout<<x<<", ";
        cout<<endl;
        **/
    }
    if(flag==false) return "";
    return result;
}

int main(){
    //string S = "ADOBECODEBANC";
    //string T = "ABC";
    //cout<<"result: "<<minWindow(S, T);
    string S = "A";
    string T = "B";
    cout<<"result: "<<minWindow(S, T);
    return 0;
}
