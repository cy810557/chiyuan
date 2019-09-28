#include<iostream>
#include<vector>
#include<string>
using namespace std;

int maxArea(vector<int> &heights){
    int max_area = 0;
    int left=0, right=heights.size()-1;
    while(left!=right){
        max_area = max(max_area, (right-left)*min(heights[left], heights[right]));
        if (heights[left] < heights[right]) ++left;
        else --right;
    }
    return max_area;
}

int main(){
    vector<int> heights = {1,8,6,2,5,4,8,3,7};
    cout<<maxArea(heights)<<endl;

}

