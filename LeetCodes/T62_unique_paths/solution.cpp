#include<vector>
#include<iostream>
int uniquePaths(int m, int n) {
    vector<int> curr(n,1);
    for(int i=1;i<m;++i){
        for(int j=1;j<n;++j)
            curr[j] += curr[j-1];
    }
    return *(curr.end()-1);
}
