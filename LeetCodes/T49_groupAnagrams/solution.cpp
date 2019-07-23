#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "../vector_utils.cpp"
using namespace std;

vector<vector<string>> group_strs(vector<string> &strs){
    /**
     * 应用哈希表将多个异位词与同一个key建立映射关系并保存到同一组。 
     **/
    vector<vector<string>> results;
    unordered_map<string, vector<string>> dicts;
    for (auto _str:strs){
        string org = _str;
        sort(_str.begin(), _str.end());
        if(dicts.find(_str)==dicts.end()) dicts[_str].push_back(org);
        else dicts[_str].push_back(org);
    }
    for (auto dict : dicts){
        results.push_back(dict.second);
    }
    return results;
}

int main()
{
    vector<vector<string>> str_groups;
    vector<string> vs = {"eat", "tea", "tan", "ate", "nat", "bat"};
    auto results = group_strs(vs);
    for(auto result : results) print(result);
    return 0;
}

