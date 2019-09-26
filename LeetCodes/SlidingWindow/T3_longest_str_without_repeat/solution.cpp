#include<iostream>
#include<string>
using namespace std;

bool is_in(char c, const string &str){
    for (auto s : str){
        if(c==s) return true;
    }
    return false;
}

int lengthOfLongestSubstring(string s) {
        int before=0, after=1;
        int count=0;
        while(after!=s.size()){
            while( & after!=s.size()){
                ++after;
                //cout<<s[before]<<" "<<s[after]<<endl;
            }
            count = max(after-before, count);
            ++before;
            after = before+1;
            cout<<before<<", "<<after<<","<<count<<endl;
        }
        return count;
}
int main(){
    string s="abcabcbb";
    cout<<s.size()<<endl;
    cout<<lengthOfLongestSubstring(s);
    
}
