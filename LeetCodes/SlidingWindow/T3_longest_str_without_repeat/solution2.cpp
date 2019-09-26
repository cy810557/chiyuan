#include<iostream>
#include<string>
#include<unordered_set>
using namespace std;

bool is_in(char c, const string &str){
    for (auto s : str){
        if(c==s) return true;
    }
    return false;
}

int lengthOfLongestSubstring(string s) {
        int left=0, count=0;
        unordered_set<char> queue;
        for (int i=0;i<s.size();++i){
            while(queue.find(s[i])!=queue.end()){
                queue.erase(s[left]);
                left++;
            }
            queue.insert(s[i]);
            count = max(count, (i-left+1));
        }
        return count;
}
int main(){
    string s="abcabcbb";
    cout<<lengthOfLongestSubstring(s);
}
